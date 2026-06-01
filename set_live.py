#!/usr/bin/env python
"""Point the landing page at the current laptop tunnel URL.

Run this each session after starting your tunnel (cloudflared/ngrok), then
commit & push docs/live.json. The printed QR never changes — only this link.

Usage:
  & "C:\\ProgramData\\anaconda3\\python.exe" set_live.py https://something.trycloudflare.com
  & "C:\\ProgramData\\anaconda3\\python.exe" set_live.py --clear      # mark laptop offline
"""
import os
import sys
import json
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_PATH = os.path.join(REPO_ROOT, "docs", "live.json")


def main():
    ap = argparse.ArgumentParser(description="Set the landing page's live (laptop) URL")
    ap.add_argument("url", nargs="?", default="", help="Tunnel URL, e.g. https://xyz.trycloudflare.com")
    ap.add_argument("--clear", action="store_true", help="Clear the live URL (laptop offline)")
    args = ap.parse_args()

    url = "" if args.clear else args.url.strip()
    os.makedirs(os.path.dirname(LIVE_PATH), exist_ok=True)
    with open(LIVE_PATH, "w", encoding="utf-8") as f:
        json.dump({"live_url": url, "note": "Set by set_live.py"}, f, indent=2)

    print(f"Wrote {LIVE_PATH}: live_url = {url or '(empty — laptop offline)'}")
    print("Now: git add docs/live.json && git commit -m 'live link' && git push")


if __name__ == "__main__":
    main()
