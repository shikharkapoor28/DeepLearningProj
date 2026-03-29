import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws/simulation/test_session"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Sending start command...")
            await websocket.send(json.dumps({"command": "start"}))
            
            print("Waiting for data payload...")
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            print("Successfully received payload!")
            print(f"Keys: {list(data.keys())}")
            print(f"Price: {data['state']['price']}")
            
            await websocket.send(json.dumps({"command": "stop"}))
            print("Test passed.")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
