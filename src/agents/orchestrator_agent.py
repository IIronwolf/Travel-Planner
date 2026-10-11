import os
import json
import re
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.schemas.query_schema import TravelQuerySchema

# Point LiteLLM / OpenAI base URL to local Ollama
os.environ["OPENAI_API_BASE"] = OLLAMA_BASE_URL
os.environ["OPENAI_API_KEY"] = "unused"

SYSTEM_INSTRUCTION = """
You are a JSON extractor for an Indian Travel AI Agent.
Convert the user query into a valid JSON object matching this exact structure:

{
  "destination_name": "Goa" or null,
  "region": "West India" or null,
  "trip_types": ["Beach", "Cultural"],
  "ideal_for": "Couple", "Solo", "Family_with_kids", "Family_with_elderly", or "Friends" (or null),
  "budget_tier": "budget", "mid_range", or "luxury",
  "max_daily_budget_inr": integer or null,
  "season": "Winter", "Summer", "Monsoon", or "Post-Monsoon" (or null),
  "duration_days": integer or null,
  "vibe_description": "summary keywords" or null
}

CRITICAL: Output ONLY valid JSON starting with '{' and ending with '}'. Do not write explanations or markdown blocks.
"""

ollama_model = LiteLlm(model=f"openai/{OLLAMA_MODEL}")

orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=ollama_model,
    instruction=SYSTEM_INSTRUCTION,
)

session_service = InMemorySessionService()
runner = Runner(
    agent=orchestrator_agent,
    app_name="travel_planner",
    session_service=session_service
)

async def _run_adk_agent(user_prompt: str) -> str:
    """Async handler to run agent via ADK Runner."""
    session = await session_service.create_session(
        app_name="travel_planner", 
        user_id="test_user"
    )
    
    content = types.Content(role="user", parts=[types.Part(text=user_prompt)])
    
    response_text = ""
    events = runner.run(
        user_id="test_user", 
        session_id=session.id, 
        new_message=content
    )
    
    for event in events:
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
                        
    return response_text


def parse_user_query(user_prompt: str) -> dict:
    """Synchronous entry point called by test_orchestrator.py."""
    raw_text = asyncio.run(_run_adk_agent(user_prompt)).strip()

    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    clean_json_str = json_match.group(0) if json_match else raw_text

    try:
        parsed_json = json.loads(clean_json_str)
        validated_schema = TravelQuerySchema(**parsed_json)
        return validated_schema.model_dump()
    except Exception as e:
        print(f"⚠️ Extraction Parsing Error: {e}")
        print(f"Raw Output Received: {raw_text}")
        return TravelQuerySchema().model_dump()