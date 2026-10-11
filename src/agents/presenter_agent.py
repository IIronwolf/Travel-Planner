import os
import json
import asyncio
from typing import Dict, Any
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL

os.environ["OPENAI_API_BASE"] = OLLAMA_BASE_URL
os.environ["OPENAI_API_KEY"] = "unused"

PRESENTER_INSTRUCTION = """
You are an expert Indian Travel Concierge.
Your job is to transform raw destination records into an engaging, structured travel itinerary.

Guidelines for formatting:
1. Provide a brief overview highlighting why this destination fits the user's travel style.
2. Present a structured day-by-day itinerary based on the duration requested and the dataset's suggested activities.
3. Include an estimated daily budget breakdown based on the matching budget tier.
4. Highlight local food recommendations, hidden gems, and best travel seasons.
5. Keep the formatting clear, clean, and easy to read using Markdown section headers and bullet points.
"""

ollama_model = LiteLlm(model=f"openai/{OLLAMA_MODEL}")

presenter_agent = LlmAgent(
    name="PresenterAgent",
    model=ollama_model,
    instruction=PRESENTER_INSTRUCTION
)

session_service = InMemorySessionService()
runner = Runner(
    agent=presenter_agent,
    app_name="travel_presenter",
    session_service=session_service
)

async def _generate_itinerary_async(user_query: str, search_results: list[Dict[str, Any]], parsed_params: Dict[str, Any]) -> str:
    """Invokes the Presenter Agent via ADK Runner."""
    session = await session_service.create_session(
        app_name="travel_presenter", 
        user_id="presenter_user"
    )

    prompt_context = f"""
    User Query: {user_query}
    Parsed User Constraints: {json.dumps(parsed_params, indent=2)}
    Matching Destination Records: {json.dumps(search_results, indent=2)}

    Please generate a complete, beautifully structured travel itinerary for the user based on these matching records.
    """

    content = types.Content(role="user", parts=[types.Part(text=prompt_context)])
    
    response_text = ""
    events = runner.run(
        user_id="presenter_user", 
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

def format_itinerary(user_query: str, search_results: list[Dict[str, Any]], parsed_params: Dict[str, Any]) -> str:
    """Synchronous wrapper for generating final travel itinerary."""
    if not search_results:
        return "No matching destinations were found for your query constraints. Try widening your budget or duration parameters."

    return asyncio.run(_generate_itinerary_async(user_query, search_results, parsed_params))