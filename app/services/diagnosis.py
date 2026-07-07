from google.genai import Client
from google.genai import types
from app.models.contracts import AgentInput, AgentOutput

# The core persona and rule engine for the diagnosis agent
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
        """Processes the input data, calls Gemini with strict schema tracking, and outputs AgentOutput."""
        
        # Build user content combining current input, active window history, and long term summary context
        prompt_content = f"Student Input: {data.current_input}\n"
        if data.running_summary:
            prompt_content = f"Historical Context of this session:\n{data.running_summary}\n\n" + prompt_content

        # Configure the request to return exact structured outputs matching our contract
        config = types.GenerateContentConfig(
            system_instruction=DIAGNOSIS_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AgentOutput,
            temperature=0.2 # Lower temperature for stable, predictable analysis
        )

        # Call Gemini natively using Google's GenAI SDK
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_content,
            config=config
        )

        # Because we supplied response_schema, the SDK automatically parses JSON into response.parsed
        # If response.parsed isn't instantly available, parse the JSON text directly into our validation contract
        if hasattr(response, 'parsed') and response.parsed:
            return response.parsed
            
        return AgentOutput.model_validate_json(response.text)