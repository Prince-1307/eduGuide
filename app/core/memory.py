import asyncio
import random
from google.genai import Client
from google.genai import types
from google.genai.errors import APIError
from app.models.contracts import ChatMessage

COMPACTION_SYSTEM_INSTRUCTION = """
You are the high-density Memory Compactor utility for eduGuide. 
Your sole responsibility is to merge an old running summary of a student session with the newest active conversation history turns.

You must output a single, highly compressed, objective bulleted summary. 
You MUST preserve these specific elements without losing them:
1. The exact topic, concept, or specific mathematical problem currently under discussion.
2. The specific active misconceptions or mistakes the student has made (e.g., "Student applied subtraction instead of division").
3. What guiding questions or specific hints the agents have already provided (so we never repeat the same hints).

Do NOT include any conversational filler, pleasantries, or introductions. Output only the raw analytical summary text.
"""

class MemoryCompactor:
    def __init__(self, client: Client, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.model_name = model_name

    async def compact_history(self, existing_summary: str, old_turns: list[ChatMessage]) -> str:
        """Compresses historical context using exponential backoff resiliency."""
        
        # Format the incoming turns into a readable text block for the prompt
        formatted_turns = ""
        for turn in old_turns:
            formatted_turns += f"{turn.role.upper()}: {turn.content}\n"

        prompt_content = (
            f"Existing Session Summary:\n{existing_summary or 'No history yet.'}\n\n"
            f"New Active Conversation Turns to Compress:\n{formatted_turns}\n"
        )

        config = types.GenerateContentConfig(
            system_instruction=COMPACTION_SYSTEM_INSTRUCTION,
            temperature=0.1,  # Ultra-low temperature for maximum deterministic compression
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
                    print(f"[Warn] Compactor hit 503/429. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise e

        return response.text.strip()