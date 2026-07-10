import re
import asyncio
import random
from google.genai import Client
from google.genai import types
from google.genai.errors import APIError # <-- Import Gemini's native error handler
from app.models.contracts import AgentInput, AgentOutput

SOCRATIC_SYSTEM_INSTRUCTION = """
You are the Socratic Homework Assistant for eduGuide. Your primary mission is to guide students toward discovering solutions on their own. 

CRITICAL LAWS OF OPERATION:
1. NEVER reveal the direct final answer, numerical result, or completed code solution to the student's specific problem under any circumstances.
2. Respond exclusively with a single, highly focused guiding question or an incremental hint that addresses their immediate point of confusion.
3. If the student is completely stuck after multiple attempts, you may provide a fully worked-out example, but it MUST use an entirely different problem statement with different numbers/variables.

Use the 'internal_analysis' field to track the student's current step and note down what hints you have already given them.
"""

class SocraticAgent:
    def __init__(self, client: Client, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.model_name = model_name

    async def execute(self, data: AgentInput) -> AgentOutput:
        """Executes Socratic guidance with resilient retry logic and code guardrails."""
        
        prompt_content = f"Student Input: {data.current_input}\n"
        if data.running_summary:
            prompt_content = f"Summary of conversation so far:\n{data.running_summary}\n\n" + prompt_content

        config = types.GenerateContentConfig(
            system_instruction=SOCRATIC_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AgentOutput,
            temperature=0.4
        )

        # RESILIENCY PARAMETERS
        MAX_RETRIES = 3
        base_delay = 1.0  # start with a 1 second delay
        response = None

        for attempt in range(MAX_RETRIES):
            try:
                # Wrap the synchronous SDK call in a thread pool to avoid blocking FastAPI's async loop
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_content,
                        config=config
                    )
                )
                break # Success! Break out of the retry loop.
            except APIError as e:
                # Catch 503 (High Demand) or 429 (Rate Limit) errors specifically
                if e.code in [503, 429] and attempt < MAX_RETRIES - 1:
                    # Calculate exponential backoff with a random jitter factor to prevent thundering herds
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = (base_delay * (2 ** attempt)) + jitter
                    print(f"[Warn] Gemini API 503/429 hit. Retrying attempt {attempt + 1} in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise e # If it's a structural error (like 400 Bad Request) or retries run out, raise it.

        # Parse out our contract schema safely
        if hasattr(response, 'parsed') and response.parsed:
            output: AgentOutput = response.parsed
        else:
            output = AgentOutput.model_validate_json(response.text)

        # DETERMINISTIC GUARDRAIL
        final_text = output.response_text.lower()
        forbidden_patterns = [r"\bx\s*=\s*\d+", r"\bans\s*=\s*\d+"]
        for pattern in forbidden_patterns:
            if re.search(pattern, final_text):
                output.response_text = "Let's pause. Look closely at your equation setup again. What operation should we perform on both sides to isolate the variable?"
                output.internal_analysis += " [BACKEND INTERCEPTED: Model attempted to reveal algebraic answer pattern.]"
                break
                
        return output