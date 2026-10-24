"""Render a talking head on your FlashHead Pro endpoint and save the mp4.

    set RUNPOD_API_KEY=...            (Windows; export on macOS/Linux)
    python examples/run_job.py --endpoint <ENDPOINT_ID> --portrait me.png --audio voice.mp3

--portrait / --audio take a local file (sent as base64), an https URL, a
/runpod-volume/... path or "sample". The job runs asynchronously (/run) and this script
polls /status, printing progress, until it finishes. A base64 result is saved to --out;
a bucket link or a network-volume path is printed.
"""
import argparse
import base64
import json
import os
import sys
import time

import requests

API = "https://api.runpod.ai/v2"


def as_input(value):
    if value == "sample" or value.startswith(("http://", "https://", "/runpod-volume/")):
        return value
    with open(value, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="endpoint ID from the RunPod console")
    ap.add_argument("--portrait", default="sample")
    ap.add_argument("--audio", default="sample")
    ap.add_argument("--out", default="flashhead.mp4")
    ap.add_argument("--anchor", default="continuous",
                    choices=["continuous", "refresh_at_pauses", "reset_every_n", "reset_every_chunk"])
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--output", default="auto", choices=["auto", "bucket", "volume", "base64"])
    args = ap.parse_args()
    key = os.environ.get("RUNPOD_API_KEY") or sys.exit("Set RUNPOD_API_KEY first.")
    headers = {"Authorization": f"Bearer {key}"}

    job = {"input": {"portrait": as_input(args.portrait), "audio": as_input(args.audio),
                     "anchor": args.anchor, "max_seconds": args.max_seconds, "output": args.output}}
    r = requests.post(f"{API}/{args.endpoint}/run", json=job, headers=headers, timeout=60)
    r.raise_for_status()
    job_id = r.json()["id"]
    print("job", job_id)

    while True:
        time.sleep(5)
        r = requests.get(f"{API}/{args.endpoint}/status/{job_id}", headers=headers, timeout=60)
        r.raise_for_status()
        status = r.json()
        state = status.get("status")
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        progress = status.get("output") if isinstance(status.get("output"), dict) else None
        print(f"  {state}" + (f"  chunk {progress['chunk']}/{progress['chunks']}" if progress and "chunk" in progress else ""),
              flush=True)

    out = status.get("output") or {}
    if state != "COMPLETED" or "error" in out:
        sys.exit(f"{state}: {out.get('error') or json.dumps(status)[:2000]}")
    if "video_base64" in out:
        with open(args.out, "wb") as f:
            f.write(base64.b64decode(out["video_base64"]))
        print("saved", args.out)
    print(json.dumps({k: v for k, v in out.items() if k != "video_base64"}, indent=1))


if __name__ == "__main__":
    main()
