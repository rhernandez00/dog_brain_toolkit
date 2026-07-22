#!/usr/bin/env python
"""Generate a printable QR code for the dashboard landing page.

The QR encodes the STABLE GitHub Pages landing URL (never the changing laptop
tunnel URL). The landing page then offers both the live and failsafe links.

Usage:
  & "C:\\ProgramData\\anaconda3\\python.exe" tools\\make_qr.py
  & "C:\\ProgramData\\anaconda3\\python.exe" tools\\make_qr.py --url https://rhernandez00.github.io/dog_brain_toolkit/

Needs the 'qrcode' package (pip install "qrcode[pil]"). If missing, prints an
online generator link you can use immediately.
"""
import os
import argparse
import webbrowser
from urllib.parse import quote

DEFAULT_URL = "https://rhernandez00.github.io/dog_brain_toolkit/"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tools/ lives one level below the repo root


def main():
    ap = argparse.ArgumentParser(description="Make a printable QR for the dashboard landing page")
    ap.add_argument("--url", default=DEFAULT_URL, help="URL to encode (the stable landing page)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "qr.png"))
    ap.add_argument("--open", action="store_true", help="Open the QR image when done")
    args = ap.parse_args()

    try:
        import qrcode
        qr = qrcode.QRCode(box_size=12, border=4)
        qr.add_data(args.url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        img.save(args.out)
        print(f"QR for: {args.url}")
        print(f"Saved : {args.out}  (print this)")
        if args.open:
            webbrowser.open(args.out)
    except ImportError:
        online = "https://api.qrserver.com/v1/create-qr-code/?size=600x600&data=" + quote(args.url, safe="")
        print("The 'qrcode' package is not installed. Two options:")
        print('  1) Install it:  pip install "qrcode[pil]"   then re-run this script.')
        print(f"  2) Use this online generator right now (downloads a PNG):\n     {online}")


if __name__ == "__main__":
    main()
