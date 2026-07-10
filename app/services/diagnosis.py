import asyncio
import random
from google.genai import Client
from google.genai import types
from google.genai.errors import APIError  # <-- Import the native error handler
from app.models.contracts import AgentInput, AgentOutput

DIAGNOSIS_SYSTEM_INSTRUCTION = """
You are the master Diagnosis Agent for eduGuide. Your sole objective is to analyze a student's incorrect mathematical or technical answer to reveal the underlying thinking process.

You must categorize the error into one of two buckets:
1. CARELESS_MISTAKE: Sign errors, simple arithmetic slips, misplaced decimals, or reading oversights where the core concept is understood.
2. CONCEPTUAL_GAP: The student fundamentally misunderstood a theorem, rule, formula application, or logic sequence.

RULES:
- You must NOT provide the actual correct numerical answer or final solution in your 'response_text'.
- Focus entirely on diagnosing the WHY. Explain the mechanics of their mistake clearly and contextually.
- Use your 'internal_analysis' field to leave technical tracking notes for the system's long-term database (e.g., specific rules misapplied).
- Keep your tone supportive, academic, and encouraging.
"""

class DiagnosisAgent:
    def __init__(self, client: Client, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.model_name = model_name

    async def execute(self, data: AgentInput) -> AgentOutput:
        """Processes the input data, calls Gemini with strict schema tracking, and outputs AgentOutput with backoff retries."""
        
        prompt_content = f"Student Input: {data.current_input}\n"
        if data.running_summary:
            prompt_content = f"Historical Context of this session:\n{data.running_summary}\n\n" + prompt_content

        config = types.GenerateContentConfig(
            system_instruction=DIAGNOSIS_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AgentOutput,
            temperature=0.2 
        )

        # RESILIENCY PARAMETERS
        MAX_RETRIES = 3
        base_delay = 1.0  
        response = None

        for attempt in range(MAX_RETRIES):
            try:
                # Thread-pool executor ensures the sync SDK call doesn't freeze the FastAPI async event loop
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_content,
                        config=config
                    )
                )
                break  # Success! Break out of the retry loop.
            except APIError as e:
                # Catch temporary high-demand (503) or rate limits (429)
                if e.code in [503, 429] and attempt < MAX_RETRIES - 1:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = (base_delay * (2 ** attempt)) + jitter
                    print(f"[Warn] Gemini API 503/429 hit in DiagnosisAgent. Retrying attempt {attempt + 1} in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise e  

        # Parse into our strict Pydantic contract
        if hasattr(response, 'parsed') and response.parsed:
            return response.parsed
            
        return AgentOutput.model_validate_json(response.text)