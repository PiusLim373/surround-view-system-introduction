#!/usr/bin/env python3
"""
discover_cameras.py

Enumerates connected V4L2 cameras, keyed by STABLE USB path (survives
replugging / renumbering / reboots) rather than the numeric /dev/videoN
index, which is not stable on this system.

For each camera, probes its advertised (format, resolution, fps)
combinations via `v4l2-ctl --list-formats-ext` and picks sane capture
settings automatically:

  - Prefer MJPG to reduce USB bandwidth for simultaneous cameras.
  - Prefer an advertised rate at or just above 30fps; never invent a rate.
  - Fall back to YUYV only if no MJPG mode meets the target.

Enumeration does not measure sustained FPS or simultaneous USB capacity.
Validate the resulting settings with pius_test.py before deploying them.

Writes the result to camera_config.json. Re-run this any time cameras
are unplugged/replugged/rebooted -- it always re-probes live hardware,
it never trusts stale assumptions about which /dev/videoN is which
camera.

Usage:
    python3 discover_cameras.py [--out camera_config.json]
"""

import argparse
import json
import os
import re
import subprocess
import sys


# Resolutions to try, in priority order, when picking MJPG fallback settings.
# Larger legroom cameras will match higher entries; bandwidth-limited ones
# will fall through to smaller ones.
MJPG_RES_PREFERENCE = [
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
    (320, 240),
]

# Minimum fps we consider "reasonable" for the MJPG fallback pick -- avoids
# picking a resolution that only offers e.g. 5fps.
TARGET_FPS = 30.0


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode:
        print(f"WARNING: {' '.join(cmd)} failed: {result.stderr.strip()}", file=sys.stderr)
    return result.stdout


def list_devices():
    """
    Parse `v4l2-ctl --list-devices` into:
        [{"name": ..., "usb_path": ..., "nodes": ["/dev/video2", ...]}, ...]
    """
    output = run(["v4l2-ctl", "--list-devices"])
    blocks = [b for b in output.split("\n\n") if b.strip()]
    cameras = []
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        header = lines[0]
        m = re.search(r"^(.*?)\s*\(([^)]+)\):\s*$", header.strip())
        if not m:
            continue
        name, usb_path = m.group(1).strip(), m.group(2).strip()
        nodes = [l.strip() for l in lines[1:] if l.strip().startswith("/dev/video")]
        if nodes:
            cameras.append({"name": name, "usb_path": usb_path, "nodes": nodes})
    return cameras


def find_stable_path(primary_node):
    """
    Given a /dev/videoN node, find the corresponding /dev/v4l/by-path/*
    symlink (stable across reboots/replugs on the same physical port).
    Returns the by-path string, or None if not found.
    """
    by_path_dir = "/dev/v4l/by-path"
    if not os.path.isdir(by_path_dir):
        return None
    target_real = os.path.realpath(primary_node)
    for entry in os.listdir(by_path_dir):
        full = os.path.join(by_path_dir, entry)
        if os.path.realpath(full) == target_real:
            return full
    return None


def parse_formats(device_node):
    """
    Parse `v4l2-ctl --device=<node> --list-formats-ext` into:
        {"MJPG": [(1920, 1080, [30.0, 25.0, 15.0]), ...], "YUYV": [...]}
    """
    output = run(["v4l2-ctl", f"--device={device_node}", "--list-formats-ext"])
    formats = {}
    current_fourcc = None
    current_res = None
    for line in output.splitlines():
        m = re.search(r"\[\d+\]:\s*'(\w+)'", line)
        if m:
            current_fourcc = m.group(1)
            current_res = None
            formats.setdefault(current_fourcc, [])
            continue
        m = re.search(r"Size:\s*Discrete\s*(\d+)x(\d+)", line)
        if m and current_fourcc:
            current_res = (int(m.group(1)), int(m.group(2)))
            formats[current_fourcc].append([current_res[0], current_res[1], []])
            continue
        m = re.search(r"\(([\d.]+)\s*fps\)", line)
        if m and current_fourcc and current_res:
            fps = float(m.group(1))
            formats[current_fourcc][-1][2].append(fps)
    return formats


def choose_settings(formats):
    """Choose only enumerated modes; advertised FPS is not a benchmark."""
    for fourcc in ("MJPG", "YUYV"):
        modes = formats.get(fourcc, [])
        ordered = sorted(modes, key=lambda mode: (
            MJPG_RES_PREFERENCE.index(tuple(mode[:2]))
            if tuple(mode[:2]) in MJPG_RES_PREFERENCE else len(MJPG_RES_PREFERENCE),
            mode[0] * mode[1]))
        for width, height, rates in ordered:
            usable = [fps for fps in rates if fps >= TARGET_FPS]
            if usable:
                return fourcc, width, height, min(usable)
    # Retain discovery information for slower cameras, but warn in main().
    for fourcc in ("MJPG", "YUYV"):
        for width, height, rates in formats.get(fourcc, []):
            if rates:
                return fourcc, width, height, max(rates)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="camera_config.json")
    args = parser.parse_args()

    cameras = list_devices()
    if not cameras:
        print("No cameras found via v4l2-ctl --list-devices", file=sys.stderr)
        sys.exit(1)

    config = {}
    print(f"{'name':<22} {'usb_path':<14} {'node':<12} {'stable_path':<45} {'chosen'}")
    print("-" * 120)

    for cam in cameras:
        primary_node = cam["nodes"][0]
        stable_path = find_stable_path(primary_node)
        formats = parse_formats(primary_node)
        chosen = choose_settings(formats)

        key = stable_path or f"usb:{cam['usb_path']}"

        if chosen is None:
            print(f"{cam['name']:<22} {cam['usb_path']:<14} {primary_node:<12} "
                  f"{str(stable_path):<45} NO USABLE FORMAT FOUND")
            continue

        fourcc, w, h, fps = chosen
        if fps < TARGET_FPS:
            print(f"WARNING: {primary_node} has no parsed MJPG/YUYV mode at >= {TARGET_FPS}fps")
        if fourcc == "YUYV":
            print(f"WARNING: {primary_node} uses uncompressed YUYV; verify shared USB bandwidth")
        chosen_str = f"{fourcc} {w}x{h} @ {fps}fps"
        print(f"{cam['name']:<22} {cam['usb_path']:<14} {primary_node:<12} "
              f"{str(stable_path):<45} {chosen_str}")

        config[key] = {
            "name": cam["name"],
            "usb_path": cam["usb_path"],
            "stable_path": stable_path,
            "last_seen_node": primary_node,
            "fourcc": fourcc,
            "width": w,
            "height": h,
            "fps": fps,
        }

    with open(args.out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nWrote {len(config)} camera configs to {args.out}")
    print("These are advertised modes, not measured rates. Run pius_test.py alone and with both cameras.")
    if any(v["stable_path"] is None for v in config.values()):
        print("WARNING: some cameras have no /dev/v4l/by-path symlink found -- "
              "falling back to usb_path as key, which is still more stable "
              "than /dev/videoN but double-check /dev/v4l/by-path exists on "
              "this system.", file=sys.stderr)


if __name__ == "__main__":
    main()
