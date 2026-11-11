#!/usr/bin/env python3
"""
make_demo_gif.py — README demo GIF from a bag's chase-camera images.

Record the run with the chase camera:
    RECORD_VIEW=1 ./scripts/reset_run.sh mppi
then cut and convert (times are seconds from the bag's first image):
    python3 scripts/make_demo_gif.py bags/run_mppi_<ts> --start 5 --end 40
    python3 scripts/make_demo_gif.py bags/run_mppi_<ts> --list     # duration and frame count only

Writes media/demo.gif by default: downscaled, frame-rate reduced and
palette-quantized (Pillow) to stay under --max-mb.  Needs rosbags and Pillow
(no ROS, no ffmpeg).  Bag timestamps are wall time; the camera runs at the
simulation's rate, so a clip plays at about the speed it was recorded at.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader

TOPIC = "/carla/ego_vehicle/rgb_view/image"


def frames(bag: Path, start: float, end: float, fps: float):
    """Yield (t, RGB array) every 1/fps s between start and end (s from the first image)."""
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        if not conns:
            sys.exit(f"{bag}: no {TOPIC} (record with RECORD_VIEW=1)")
        first = next_t = None
        for conn, stamp, raw in reader.messages(connections=conns):
            t = stamp * 1e-9
            if first is None:
                first = t
                next_t = first + start
            if t < next_t:
                continue
            if t > first + end:
                break
            next_t += 1.0 / fps
            msg = reader.deserialize(raw, conn.msgtype)
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding.lower().startswith("bgr"):
                img = img[:, :, 2::-1]          # BGR(A) → RGB
            yield t - first, np.ascontiguousarray(img[:, :, :3])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="s from the first image")
    ap.add_argument("--end", type=float, default=40.0, help="s from the first image")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--max-mb", type=float, default=8.0)
    ap.add_argument("-o", "--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / "media/demo.gif")
    ap.add_argument("--list", action="store_true", help="print the clip's length and frame count")
    args = ap.parse_args()

    if args.list:
        n, last = 0, 0.0
        for t, _ in frames(args.bag, 0.0, 1e9, 1e9):
            n, last = n + 1, t
        print(f"{n} images over {last:.1f} s")
        return

    width = args.width
    while True:
        imgs = []
        for _, rgb in frames(args.bag, args.start, args.end, args.fps):
            im = Image.fromarray(rgb)
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            imgs.append(im.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE))
        if not imgs:
            sys.exit("no images in that window")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(args.output, save_all=True, append_images=imgs[1:],
                     duration=round(1000 / args.fps), loop=0, optimize=True)
        mb = args.output.stat().st_size / 1e6
        print(f"{args.output}: {len(imgs)} frames, {width} px wide, {mb:.1f} MB")
        if mb <= args.max_mb or width <= 240:
            break
        width = int(width * 0.8)
        print(f"over {args.max_mb} MB; retrying at {width} px")


if __name__ == "__main__":
    main()
