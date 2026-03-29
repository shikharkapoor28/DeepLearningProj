import asyncio
from main import run_live_simulation

async def test():
    try:
        await run_live_simulation("test_session")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import main
    main.is_simulating = True
    asyncio.run(test())
