#!/usr/bin/env python3
"""Profile infernix-serve using nsys and collect all kernel names."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

def wait_for_server(url="http://127.0.0.1:8000/v1/models", timeout=120):
    start = time.time()
    print(f"Waiting for server at {url} to become ready...")
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    model_id = data["data"][0]["id"]
                    print(f"Server is ready! Registered Model ID: '{model_id}'")
                    return model_id
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Server did not become ready within {timeout} seconds")

def send_request(model_id, prompt="Explain the architecture of NVIDIA DGX Spark GB10 in 3 concise bullet points.", max_tokens=128):
    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    print(f"Sending prompt to {url} with model='{model_id}'...")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode())
        dt = time.time() - t0
        content = res["choices"][0]["message"]["content"]
        usage = res.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        speed = tokens / max(dt, 0.001)
        print(f"Request completed in {dt:.2f}s ({tokens} tokens, {speed:.1f} tok/s)")
        print(f"\nResponse excerpt:\n{content[:200]}...\n")
        return res

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_rep = os.path.join(repo_root, "profiles", "nsys", "server_kernels")
    rep_file = f"{output_rep}.nsys-rep"
    
    # Clean previous reports
    for ext in [".nsys-rep", ".sqlite"]:
        f = f"{output_rep}{ext}"
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    env = os.environ.copy()
    env["SKIP_BUILD"] = "1"
    
    cmd = [
        "nsys", "profile",
        "-t", "cuda,nvtx",
        "-o", output_rep,
        "--force-overwrite=true",
        "bash", os.path.join(repo_root, "tools", "run_server.sh"), "hc4"
    ]
    
    print(f"Launching nsys profile: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    
    try:
        model_id = wait_for_server(timeout=120)
        
        # Send requests (short decode and prefill)
        print("\n--- Sending Request 1 ---")
        send_request(model_id, "What is Infernix C++ inference engine? Describe its architecture in 2 sentences.", max_tokens=64)
        
        print("\n--- Sending Request 2 ---")
        send_request(model_id, "List 3 high performance CUDA optimization techniques for unified memory systems.", max_tokens=64)
        
        print("\nRequests completed successfully. Stopping nsys session gracefully...")
    finally:
        # Send SIGTERM to the process group so infernix-serve terminates and nsys generates the report
        print(f"Sending SIGTERM to process group {proc.pid}...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError as e:
            print(f"killpg error: {e}")
            
        try:
            proc.wait(timeout=45)
            print("nsys process exited cleanly.")
        except subprocess.TimeoutExpired:
            print("nsys process timed out waiting for exit, sending SIGKILL...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
            except Exception as e:
                print(f"killpg SIGKILL error: {e}")

    # Check if .nsys-rep was produced
    if not os.path.exists(rep_file):
        print(f"Error: {rep_file} was not generated!", file=sys.stderr)
        return 1

    print(f"\nReport generated at {rep_file}")
    print("\n--- Extracting CUDA GPU Kernel Summary via nsys stats ---")
    stats_cmd = [
        "nsys", "stats",
        "--report", "cuda_gpu_kern_sum",
        "--format", "csv",
        rep_file
    ]
    
    result = subprocess.run(stats_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"nsys stats failed:\n{result.stderr}", file=sys.stderr)
        return 1
        
    print(result.stdout)
    
    # Save the output CSV for analysis
    csv_file = os.path.join(repo_root, "profiles", "nsys", "kernel_summary.csv")
    with open(csv_file, "w") as f:
        f.write(result.stdout)
    print(f"Saved kernel summary CSV to {csv_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
