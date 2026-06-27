#!/usr/bin/env python3
"""Compatibility entry point for interactive 3DGS visualization.

This module used to export headless renders to image files. It now delegates
to the viser-based viewer so remote inspection can be done in a browser.
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.gaussian.viewer import make_viewer_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive 3DGS viewer (replaces headless image export)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the viewer server")
    parser.add_argument("--width", type=int, default=768, help="Render width")
    parser.add_argument("--height", type=int, default=432, help="Render height")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    viewer = make_viewer_from_checkpoint(
        checkpoint_path=args.checkpoint,
        width=args.width,
        height=args.height,
        port=args.port,
        host=args.host,
        device=args.device,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
