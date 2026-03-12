import websocket
import requests
import json
import uuid

# --- CONFIG ---
SERVER_ADDRESS = "13.193.97.70:8188"
CLIENT_ID = str(uuid.uuid4())

def test_connection():
    print(f"--- Starting Connection Test to {SERVER_ADDRESS} ---")
    
    # 1. Test HTTP API (The "Front Door")
    try:
        print(f"1. Testing HTTP API...")
        response = requests.get(f"http://{SERVER_ADDRESS}/object_info", timeout=5)
        if response.status_code == 200:
            print("   ✅ HTTP API is reachable.")
        else:
            print(f"   ❌ HTTP API returned status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ HTTP Connection Failed: {e}")
        return

    # 2. Test WebSocket (The "Live Feed")
    try:
        print(f"2. Testing WebSocket Handshake...")
        ws = websocket.create_connection(f"ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}", timeout=5)
        
        # Send a small 'ping' or just wait for the initial 'status' message
        result = ws.recv()
        data = json.loads(result)
        
        if data.get('type') == 'status':
            print("   ✅ WebSocket Handshake Successful.")
            print("   ✅ EC2 sent status: Server is IDLE and ready.")
        else:
            print(f"   ⚠️ WebSocket connected but received unexpected data: {data['type']}")
        
        ws.close()
    except Exception as e:
        print(f"   ❌ WebSocket Connection Failed: {e}")

if __name__ == "__main__":
    test_connection()