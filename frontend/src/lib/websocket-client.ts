import { SimulationPayload } from "@/types/stream";

type MessageHandler = (data: SimulationPayload) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private onMessageHandlers: Set<MessageHandler> = new Set();
  private reconnectAttempt = 0;
  
  public isConnected = false;

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.isConnected = true;
      this.reconnectAttempt = 0;
      console.log("WebSocket connected to", this.url);
    };

    this.ws.onmessage = (event) => {
      try {
        const data: SimulationPayload = JSON.parse(event.data);
        this.onMessageHandlers.forEach(handler => handler(data));
      } catch (err) {
        console.error("Failed to parse WebSocket message", err);
      }
    };

    this.ws.onclose = () => {
      this.isConnected = false;
      this.scheduleReconnect();
    };

    this.ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      this.ws?.close();
    };
  }

  subscribe(handler: MessageHandler) {
    this.onMessageHandlers.add(handler);
    return () => this.onMessageHandlers.delete(handler);
  }

  send(command: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(command));
    }
  }

  disconnect() {
    this.ws?.close();
  }

  private scheduleReconnect() {
    const delay = Math.min(1000 * (2 ** this.reconnectAttempt), 10000);
    this.reconnectAttempt++;
    setTimeout(() => this.connect(), delay);
  }
}
