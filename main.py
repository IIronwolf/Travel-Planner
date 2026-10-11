from src.agents.orchestrator_agent import parse_user_query
from src.tools.search_tools import hybrid_search
from src.agents.presenter_agent import format_itinerary

def run_travel_agent(user_prompt: str):
    print("\n" + "=" * 60)
    print(f"✈️ USER QUERY: {user_prompt}")
    print("=" * 60)

    print("\n[1/3] Parsing query constraints with Orchestrator Agent...")
    extracted_params = parse_user_query(user_prompt)
    print(f"✓ Target Destination: {extracted_params.get('destination_name') or 'Flexible'}")
    print(f"✓ Trip Duration: {extracted_params.get('duration_days')} Days")
    print(f"✓ Budget Tier: {extracted_params.get('budget_tier')}")

    print("\n[2/3] Querying SQLite and ChromaDB vector store...")
    search_results = hybrid_search(extracted_params, top_k=2)
    print(f"✓ Found {len(search_results)} matching destination record(s).")

    print("\n[3/3] Formulating custom itinerary with Presenter Agent...\n")
    final_output = format_itinerary(user_prompt, search_results, extracted_params)
    
    print("-" * 60)
    print(final_output)
    print("-" * 60)


if __name__ == "__main__":
    print("Welcome to Offline-First Indian Travel Concierge!")
    
    while True:
        try:
            query = input("\nEnter your travel request (or 'exit' to quit):\n> ")
            if query.strip().lower() in ["exit", "quit", "q"]:
                break
            if not query.strip():
                continue
            
            run_travel_agent(query)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ An error occurred: {e}")