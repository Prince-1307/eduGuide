import asyncio
import random
from google.genai import Client
from google.genai import types
from google.genai.errors import APIError
from app.models.contracts import AgentInput, AgentOutput

REGIONAL_SYSTEM_INSTRUCTION = """
You are the Regional-Language Study Companion for eduGuide. Your primary mission is to break down complex academic or technical terms into a student's preferred local language.

CRITICAL LAWS OF OPERATION:
1. Identify the target language from the request context (default to Hindi if not specified).
2. Translate the core academic definition accurately into the target language script (e.g., Devanagari for Hindi, Gujarati script, etc.).
3. Crucially, you MUST provide a localized, everyday cultural analogy common to the region (e.g., Indian households, local transport, popular sports) to make the abstract concept instantly clear.
4. Keep the explanation supportive, simple, and avoid overly formal or archaic dictionary words that make the language harder to read than the English original.

Use the 'internal_analysis' field to document the exact linguistic translation choice and key cultural elements chosen for the analogy.
"""

class RegionalAgent:
    def __init__(self, client: Client, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.model_name = model_name

    async def execute(self, data: AgentInput) -> AgentOutput:
        """Executes regional translation and localized explanation with network resiliency."""
        
        # Pull target language dynamically from our metadata contract dictionary
        target_lang = data.metadata.get("preferred_language", "Hindi")
        
        prompt_content = (
            f"Target Language: {target_lang}\n"
            f"Concept to explain: {data.current_input}\n"
        )
        
        if data.running_summary:
            prompt_content = f"Session Context:\n{data.running_summary}\n\n" + prompt_content

        config = types.GenerateContentConfig(
            system_instruction=REGIONAL_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AgentOutput,
            temperature=0.3  # Lower temperature for stable translation grounding
        )

        MAX_RETRIES = 3
        base_delay = 1.0
        response = None

        for attempt in range(MAX_RETRIES):
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_content,
                        config=config
                    )
                )
                break
            except APIError as e:
                if e.code in [503, 429] and attempt < MAX_RETRIES - 1:
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = (base_delay * (2 ** attempt)) + jitter
                    print(f"[Warn] Gemini API 503/429 hit in RegionalAgent. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise e

        if hasattr(response, 'parsed') and response.parsed:
            return response.parsed
            
        return AgentOutput.model_validate_json(response.text)