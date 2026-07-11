import os
import asyncio
from fastapi import FastAPI, HTTPException
from google.genai import Client
from google.genai import types
from dotenv import load_dotenv

from app.models.contracts import AgentInput, AgentOutput, ChatMessage
from app.services.diagnosis import DiagnosisAgent
from app.services.socratic import SocraticAgent
from app.services.regional import RegionalAgent
from app.core.memory import MemoryCompactor

load_dotenv()

app = FastAPI(title="eduGuide Unified Multi-Agent Orchestrator")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")

# Shared client resource across all units
gemini_client = Client()

# Instantiate all specialized modules
diagnosis_agent = DiagnosisAgent(client=gemini_client)
socratic_agent = SocraticAgent(client=gemini_client)
regional_agent = RegionalAgent(client=gemini_client)
memory_compactor = MemoryCompactor(client=gemini_client)

MAX_BUFFER_LIMIT = 6

ROUTER_SYSTEM_INSTRUCTION = """
You are the central Routing Router for eduGuide. Your job is to classify a student's input message into exactly one of three target agent categories:
1. DIANOSTIC: The student is providing an answer they got wrong, complaining about why an answer is wrong, or trying to figure out a specific error.
2. SOCRATIC: The student is asking for direct homework answers, requesting help solving a specific problem statement from scratch, or asking for hints.
3. REGIONAL: The student is asking to translate a topic, explain an abstract definition in a specific local language, or wants a simpler breakdown of a concept.

Output ONLY a single plain text word matching the category uppercase name: 'DIAGNOSTIC', 'SOCRATIC', or 'REGIONAL'. Do not include markdown formatting or punctuation.
"""

async def determine_route(current_input: str) -> str:
    """Uses a fast semantic evaluation call to classify student intent."""
    config = types.GenerateContentConfig(
        system_instruction=ROUTER_SYSTEM_INSTRUCTION,
        temperature=0.0
    )
    # Use thread pool to keep the FastAPI event loop unblocked
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=current_input,
            config=config
        )
    )
    return response.text.strip().upper()

@app.post("/api/chat", response_model=AgentOutput)
async def unified_chat_orchestrator(payload: AgentInput):
    """
    Main End-to-End Orchestrator Endpoint.
    Manages Summarized Memory and routes dynamically to the correct specialized agent.
    """
    try:
        # 1. Check if Short-Term Buffer needs background compaction
        if len(payload.active_history) >= MAX_BUFFER_LIMIT:
            print(f"[Core] History limit ({len(payload.active_history)}) reached. Initiating memory compaction...")
            
            # Extract oldest 4 turns to compress, leaving the newest turns intact
            turns_to_compact = payload.active_history[:4]
            payload.active_history = payload.active_history[4:]
            
            # Run the background compactor to update the running summary
            updated_summary = await memory_compactor.compact_history(
                existing_summary=payload.running_summary,
                old_turns=turns_to_compact
            )
            payload.running_summary = updated_summary
            print("[Core] Memory compaction completed successfully.")

        # 2. Determine target agent route dynamically
        target_route = await determine_route(payload.current_input)
        print(f"[Router] Routing request semantically to: {target_route}")

        # 3. Direct execution payload to target isolated agent
        if "DIAGNOSTIC" in target_route:
            response = await diagnosis_agent.execute(payload)
        elif "REGIONAL" in target_route or payload.metadata.get("preferred_language") and "English" not in payload.metadata.get("preferred_language", ""):
            response = await regional_agent.execute(payload)
        else:
            # Fallback to Socratic Homework Assistant
            response = await socratic_agent.execute(payload)

        # 4. Inject updated memory states back into the contract response 
        # so the client application layer knows the new running state value
        response.internal_analysis += f" | [Session Summary: {payload.running_summary}]"
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration Error: {str(e)}")