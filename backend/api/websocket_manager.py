from fastapi import WebSocket
from typing import Dict, List
import json

class ConnectionManager:
    """Manages active WebSockets for streaming simulation data to the frontend."""
    
    def __init__(self):
        # Maps session_id to active connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            # Serialize once for all clients in session
            payload = json.dumps(message)
            for connection in self.active_connections[session_id]:
                await connection.send_text(payload)
