import os
from fastapi import FastAPI, HTTPException
from google.genai import Client
from dotenv import load_dotenv
from app.models.contracts import AgentInput, AgentOutput
from app.services.diagnosis import DiagnosisAgent

app = FastAPI(title="eduGuide AI System Core")

load_dotenv()

# Initialize the Gemini Client using the new google-genai SDK spec
# It automatically picks up the GEMINI_API_KEY environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")

gemini_client = Client()

# Instantiate our Diagnosis Agent
diagnosis_agent = DiagnosisAgent(client=gemini_client)

@app.get("/")
def read_root():
    return {"status": "healthy", "system": "eduGuide Core Orchestrator"}

@app.post("/api/agent/diagnosis", response_model=AgentOutput)
async def process_diagnosis(payload: AgentInput):
    """
    Direct endpoint to test the Diagnosis Agent in isolation.
    Validates data through Pydantic contracts automatically.
    """
    try:
        response = await diagnosis_agent.execute(payload)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Execution Error: {str(e)}")