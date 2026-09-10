import json
import sys
from client import GradientClipper

clipper = GradientClipper()

def handle_rpc(line):
    try:
        req = json.loads(line)
        method = req.get("method")
        params = req.get("params", {})
        rid = req.get("id")
        
        if method == "tools/list":
            tools = [
                {"name": "clip_grads", "description": "Clip gradient list by global L2 norm"}
            ]
            return json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"tools": tools}})
        elif method == "tools/call":
            tname = params.get("name")
            args = params.get("arguments", {})
            if tname == "clip_grads":
                g, n, c = clipper.clip_gradients(args["gradients"])
                return json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"clipped_grads": g, "norm": n, "was_clipped": c}})
    except Exception as e:
        return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}})

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip():
            print(handle_rpc(line.strip()), flush=True)
