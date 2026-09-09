"""
camera_config_loader.py

Loads camera_config.json (produced by discover_cameras.py) and resolves
each camera's STABLE identity to whatever /dev/videoN it currently is,
at the moment the script runs. This is the piece that makes replugging
"just work" -- the JSON never stores a device number, only a stable
path/usb_path plus the chosen capture settings; the actual /dev/videoN
integer is looked up fresh every run.
"""

import json
import os
import re


def load_camera_config(path="camera_config.json"):
    with open(path) as f:
        return json.load(f)


def resolve_device_index(cam_entry):
    """
    Given one camera's config entry (as stored in camera_config.json),
    resolve it to the CURRENT /dev/videoN integer index.

    Resolution order:
      1. stable_path (/dev/v4l/by-path/...) if it still exists -- this is
         the reliable path, tied to physical USB port.
      2. last_seen_node, as a last-resort fallback if the by-path symlink
         is unavailable (e.g. udev rule not installed on this system).
         Not guaranteed correct after a replug -- prints a warning.

    Returns an int device index suitable for CaptureThread(device_id=...),
    or None if it can't be resolved (camera not currently connected).
    """
    stable_path = cam_entry.get("stable_path")
    if stable_path and os.path.exists(stable_path):
        real = os.path.realpath(stable_path)
        m = re.search(r"/dev/video(\d+)", real)
        if m:
            return int(m.group(1))

    last_seen = cam_entry.get("last_seen_node")
    if last_seen and os.path.exists(last_seen):
        print(f"WARNING: no stable by-path symlink for {cam_entry.get('name')}, "
              f"falling back to last_seen_node={last_seen}. This may be WRONG "
              f"if devices have been replugged since discovery -- re-run "
              f"discover_cameras.py to refresh.")
        m = re.search(r"/dev/video(\d+)", last_seen)
        if m:
            return int(m.group(1))

    return None


def resolve_all(config):
    """
    Returns a list of dicts, one per camera, each with:
        device_id, fourcc, resolution (w,h), fps, name
    Cameras that can't currently be resolved (unplugged) are skipped,
    with a printed warning.
    """
    resolved = []
    seen = set()
    for key, entry in config.items():
        device_id = resolve_device_index(entry)
        if device_id is None:
            print(f"WARNING: camera '{entry.get('name')}' ({key}) not found "
                  f"-- is it plugged in? Skipping.")
            continue
        if device_id in seen:
            raise ValueError(f"Multiple config entries resolve to /dev/video{device_id}")
        seen.add(device_id)
        stable_path = entry.get("stable_path")
        resolved.append({
            "device_id": device_id,
            "device_path": (stable_path if stable_path and os.path.exists(stable_path)
                            else f"/dev/video{device_id}"),
            "fourcc": entry["fourcc"],
            "resolution": (entry["width"], entry["height"]),
            "fps": entry["fps"],
            "name": entry["name"],
        })
    return resolved
