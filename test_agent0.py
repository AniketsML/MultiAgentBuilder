import asyncio
from backend.agents.agent0_extractor import extract_states

async def main():
    context_doc = """
    We want to build a customer support bot for our e-commerce store. 
    It should handle user greetings, ask for their order number to check order status, 
    and also allow them to request a refund. If the user is very angry, hand off to a human agent.
    """
    try:
        print("Running Agent 0: State Extractor...")
        result = await extract_states(context_doc)
        import json
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error running agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())
