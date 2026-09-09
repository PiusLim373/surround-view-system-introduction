#!/usr/bin/env python3

import json
import os
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from surround_view import CaptureThread, CameraProcessingThread
from surround_view import FisheyeCameraModel
# from surround_view import BirdView
from surround_view import MultiBufferManager, ProjectedImageBuffer
from surround_view.two_camera_view import TwoCameraView
import camera_config_loader as ccl
# import surround_view.param_settings as settings

# --- Wrist / third-person cam config ---
TARGET_HEIGHT = 480
JPEG_QUALITY  = 50

CAMERAS = [
    # {"device": "/dev/wrist_cam",         "topic": "/wrist_cam/compressed"},
    # {"device": "/dev/third_person_cam",  "topic": "/third_person_cam/compressed", "rotate": cv2.ROTATE_180},
]

# how often a disconnected camera retries opening its device node
RECONNECT_INTERVAL_SEC = 2.0
# how often the surround-view watchdog checks pipeline health / retries a rebuild
RECONNECT_CHECK_INTERVAL_SEC = 2.0

# Keep the published topic/size compatible with the existing consumer while
# limiting encode/publish work. The two projected feeds are stacked vertically.
OUTPUT_WIDTH = 780
OUTPUT_HEIGHT = 1040
OUTPUT_FPS = 30
OUTPUT_JPEG_QUALITY = 50

# Display-only controls; capture settings and camera YAMLs remain independent.
DOG_DISPLAY_HEIGHT = 150
VIEW_PERIPHERAL_WIDTH = 1.15  # 1.0 = original projected width; larger = wider view
VIEW_EDGE_FEATHER = 28       # pixels: soften the actual camera coverage boundary
VIEW_NEAR_FADE = 0           # keep ground beside the dog's nose/tail fully visible
VIEW_CONE_HALF_ANGLE = 72     # increase to expose wider near-field side coverage
VIEW_SHOW_LABELS = True

# --- Surround view config ---
repo_dir            = os.path.dirname(os.path.abspath(__file__))
yamls_dir           = os.path.join(repo_dir, "yaml")
camera_config_path  = os.path.join(repo_dir, "camera_config.json")

# Lightweight compositor: only these two cameras are opened and processed.
camera_ids     = ["/dev/front_cam", "/dev/back_cam"]
flip_methods   = [0, 0]
names          = ["front", "back"]
cameras_files  = [os.path.join(yamls_dir, name + ".yaml") for name in names]
camera_models  = [FisheyeCameraModel(cf, n) for cf, n in zip(cameras_files, names)]

dog_image = cv2.imread(os.path.join(repo_dir, "images", "dog.png"), cv2.IMREAD_UNCHANGED)
view_renderer = TwoCameraView(
    camera_models,
    dog_image,
    width=OUTPUT_WIDTH,
    height=OUTPUT_HEIGHT,
    dog_height=DOG_DISPLAY_HEIGHT,
    peripheral_width=VIEW_PERIPHERAL_WIDTH,
    edge_feather=VIEW_EDGE_FEATHER,
    near_fade=VIEW_NEAR_FADE,
    cone_half_angle=VIEW_CONE_HALF_ANGLE,
    show_labels=VIEW_SHOW_LABELS)

# Device identity stays pinned to the front_cam/back_cam udev symlinks above,
# regardless of /dev/videoN renumbering, so camera_config.json is NOT used to
# pick device paths. It is used for per-camera capture metadata keyed by USB
# stable_path, as camera_config_loader.py expects. Edit these keys to match your
# wiring; cross-check with `v4l2-ctl --list-devices` or /dev/v4l/by-path.
POSITION_TO_CONFIG_KEY = {
    "front": "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.4.4:1.0-video-index0",
    "back":  "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.4.1:1.0-video-index0",
}

# Fallback capture settings for any position missing from camera_config.json
# (or if the file itself fails to load) so build_pipeline() never breaks.
DEFAULT_CAPTURE_SETTINGS = {"resolution": (640, 480), "fps": 30, "fourcc": "MJPG"}


def _make_offline_frame(width, height, label):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(width, height) / 400.0)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(0, (width - tw) // 2)
    y = max(th, (height + th) // 2)
    cv2.putText(frame, label, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return frame


def resolve_capture_settings():
    """Re-read camera_config.json and pull each enabled mount position's
    (front/back, from `names`) resolution, fps and pixel format.

    Device IDENTITY is intentionally NOT touched here -- camera_ids stays
    the hardcoded /dev/front_cam-style symlink list above, unaffected by
    what /dev/videoN a camera currently enumerates as. This function only
    supplies settings, keyed by the same USB stable_path used in
    POSITION_TO_CONFIG_KEY / camera_config_loader.py.

    Called fresh on every build_pipeline() attempt (not once at import
    time), so edits to camera_config.json (e.g. after re-running
    discover_cameras.py) are picked up on the next pipeline rebuild without
    restarting run_tmms_2cam.py.

    Returns a list of resolution/fps/fourcc dicts in `names`
    order. Any position missing from the file, or if the file fails to
    load entirely, falls back to DEFAULT_CAPTURE_SETTINGS with a warning.
    """
    try:
        config = ccl.load_camera_config(camera_config_path)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[surround] failed to read {camera_config_path}: {e}, "
              f"using default capture settings for all cameras")
        config = {}

    settings_list = []
    for pos in names:
        key = POSITION_TO_CONFIG_KEY.get(pos)
        entry = config.get(key) if key else None
        if entry is not None:
            settings_list.append({
                "resolution": (entry["width"], entry["height"]),
                "fps": entry["fps"],
                "fourcc": entry.get("fourcc", "MJPG"),
            })
        else:
            print(f"WARNING: no camera_config.json entry for '{pos}' "
                  f"-- using default capture settings {DEFAULT_CAPTURE_SETTINGS}")
            settings_list.append(dict(DEFAULT_CAPTURE_SETTINGS))
    return settings_list


def camera_thread(node, pub, device_path, stop_event, rotate=None):
    label = device_path.rsplit("/", 1)[-1]
    offline_frame = _make_offline_frame(640, TARGET_HEIGHT, f"{label} OFFLINE")

    def publish(frame):
        msg              = CompressedImage()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.format       = "jpeg"
        _, buf           = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        msg.data         = buf.tobytes()
        pub.publish(msg)

    cap = None
    connected = False
    last_attempt = 0.0

    while not stop_event.is_set():
        if not connected:
            now = time.monotonic()
            if now - last_attempt >= RECONNECT_INTERVAL_SEC:
                last_attempt = now
                if os.path.exists(device_path):
                    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 15)
                        connected = True
                        print(f"[{label}] connected")
                    else:
                        cap.release()
                        cap = None

            if not connected:
                publish(offline_frame)
                time.sleep(0.2)
                continue

        ret, frame = cap.read()
        if not ret:
            print(f"[{label}] read failed, marking offline")
            cap.release()
            cap = None
            connected = False
            continue

        if rotate is not None:
            frame = cv2.rotate(frame, rotate)

        fh, fw = frame.shape[:2]
        new_w  = int(fw * TARGET_HEIGHT / fh)
        frame  = cv2.resize(frame, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        publish(frame)

    if cap is not None:
        cap.release()
    print(f"[{label}] closed")


def build_pipeline():
    """Construct and start one fresh generation of the 2-camera compositor.
    Device paths are the static front_cam/back_cam symlinks; capture settings
    (resolution, fps, fourcc) are re-read from camera_config.json on every call (see
    resolve_capture_settings()) so edits to that file take effect on the
    next pipeline rebuild without restarting run_tmms.py.

    If either camera cannot be opened, the watchdog retries the complete pair
    after RECONNECT_CHECK_INTERVAL_SEC.
    """
    capture_settings = resolve_capture_settings()

    for pos, cid, cm, cs in zip(names, camera_ids, camera_models, capture_settings):
        expected = POSITION_TO_CONFIG_KEY[pos]
        print(f"[surround] {pos}: {cid} -> {os.path.realpath(cid)}; config port: {expected}")
        if os.path.exists(cid) and os.path.exists(expected) and not os.path.samefile(cid, expected):
            print(f"[surround] ERROR: {cid} points to a different camera than {expected}; fix udev mapping")
            return None
        if tuple(cm.resolution) != tuple(cs["resolution"]):
            print(f"[surround] WARNING: {pos} calibration is {tuple(cm.resolution)}, "
                  f"capture is {cs['resolution']}; projection requires matching calibration")
    if all(os.path.exists(cid) for cid in camera_ids) and os.path.samefile(*camera_ids):
        print("[surround] ERROR: front/back aliases point to the same device")
        return None

    capture_tds = [
        CaptureThread(cid, fm, resolution=cs["resolution"],
                      fourcc=cs["fourcc"], frps=cs["fps"], use_gst=False)
        for cid, fm, cs in zip(camera_ids, flip_methods, capture_settings)
    ]
    capture_buffer_manager = MultiBufferManager(do_sync=False)
    for td in capture_tds:
        # A short queue minimizes latency if projection briefly falls behind.
        capture_buffer_manager.bind_thread(td, buffer_size=2, sync=False)
        if not td.connect_camera():
            # print(f"[capture] {td.device_id} could not be opened -- will use black placeholder")
            for t2 in capture_tds:
                t2.disconnect_camera()
            return None

    for td in capture_tds:
        td.start()

    proc_buffer_manager = ProjectedImageBuffer(buffer_size=2, do_sync=False)
    process_tds = [
        CameraProcessingThread(capture_buffer_manager, cid, cm,
                               frame_transform=view_renderer.projectors[cm.camera_name].project)
        for cid, cm in zip(camera_ids, camera_models)
    ]
    for td in process_tds:
        proc_buffer_manager.bind_thread(td)
    for td in process_tds:
        td.start()

    return dict(
        capture_tds=capture_tds,
        capture_buffer_manager=capture_buffer_manager,
        process_tds=process_tds,
        proc_buffer_manager=proc_buffer_manager,
    )


def stop_pipeline(gen):
    """Best-effort teardown using only the surround_view package's public API.
    Both CaptureThreads stop and release their cameras. Projector threads are
    released through ProjectedImageBuffer.wake_all().
    """
    for td in gen["capture_tds"]:
        td.stop()
    for td in gen["process_tds"]:
        td.stop()
    gen["proc_buffer_manager"].wake_all()
    for td in gen["capture_tds"] + gen["process_tds"]:
        if not td.wait(2000):
            print(f"[surround] waiting for {td.device_id} to finish before rebuilding")
            td.wait()


def pipeline_is_healthy(gen, last_counts):
    """Frame-count-progress check per capture thread. More reliable than
    is_camera_connected()/isOpened(), which can stay stale-True after a
    physical unplug.

    NOTE: CaptureThread.run() (capture_thread.py) falls back to pushing
    black placeholder frames -- and still increments frames_processed_count
    -- once a camera has failed to grab() for longer than
    CaptureThread.OFFLINE_TIMEOUT_SEC. So this counter means "is this
    device's buffer being fed at all (real OR placeholder)", not strictly
    "is this camera successfully grabbing real frames".

    CaptureThread retries its stable device path while offline. A placeholder
    keeps processing alive but is never counted in the real FPS diagnostics.

    A device seen for the first time gets one free grace tick."""
    healthy = True
    for td in gen["capture_tds"]:
        diagnostic = td.diagnostics()
        sample_key = (td.device_id, "diagnostic")
        before = last_counts.get(sample_key)
        if before is not None:
            elapsed = diagnostic["sample_time"] - before["sample_time"]
            real_fps = (diagnostic["real_frames"] - before["real_frames"]) / max(elapsed, 1e-6)
            print(f"[surround {td.device_id}] real_fps={real_fps:.2f} "
                  f"placeholders={diagnostic['placeholder_frames']} "
                  f"grab_fail={diagnostic['grab_failures']} "
                  f"decode_fail={diagnostic['retrieve_failures']} "
                  f"last_real_age={diagnostic['last_real_age_sec']}")
        last_counts[sample_key] = diagnostic
        prev = last_counts.get(td.device_id, -1)
        cur = td.stat_data.frames_processed_count
        if cur == prev:
            healthy = False
        last_counts[td.device_id] = cur
    return healthy


def surround_watchdog(state, stop_event):
    last_counts = {}
    while not stop_event.is_set():
        with state["lock"]:
            gen = state["pipeline"]

        if gen is None:
            new_gen = build_pipeline()
            if new_gen is not None:
                print("[surround] pipeline online")
                last_counts = {}
                with state["lock"]:
                    state["pipeline"] = new_gen
        elif not pipeline_is_healthy(gen, last_counts):
            print("[surround] camera stalled, tearing down pipeline")
            with state["lock"]:
                state["pipeline"] = None
            stop_pipeline(gen)

        time.sleep(RECONNECT_CHECK_INTERVAL_SEC)


def compose_two_camera_view(frames):
    """Blend independent front/rear cone layers around the robot icon."""
    return view_renderer.compose({name: frames.get(cid) for name, cid in zip(names, camera_ids)})


def topdown_thread(node, pub, state, stop_event):
    offline_frame   = _make_offline_frame(
        OUTPUT_WIDTH, OUTPUT_HEIGHT, "2-CAMERA VIEW OFFLINE")
    last_good_frame = None
    last_gen        = None
    frame_period    = 1.0 / OUTPUT_FPS
    stats_started = time.monotonic()
    published = 0
    previous_processed = {}
    print("[topdown] 2-camera compositor publishing started")
    while not stop_event.is_set():
        loop_started = time.monotonic()
        with state["lock"]:
            gen = state["pipeline"]

        # a fresh generation hasn't produced anything yet -- don't reuse a
        # frame left over from a previous (possibly abandoned) generation
        if gen is not last_gen:
            last_good_frame = None
            last_gen = gen
            previous_processed = {}

        if gen is not None:
            # Drain the short queue and compose its newest snapshot. Cameras
            # update independently; this is not hardware frame synchronization.
            frames = None
            for _ in range(gen["proc_buffer_manager"].buffer.size()):
                frames = gen["proc_buffer_manager"].get()
            if frames is not None:
                last_good_frame = compose_two_camera_view(frames)
            img = last_good_frame if last_good_frame is not None else offline_frame
        else:
            img = offline_frame

        # uncomment this to see the topdown view in a window on the host machine (requires X11)
        # cv2.imshow("topdown", img)
        # cv2.waitKey(1)

        msg              = CompressedImage()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.format       = "jpeg"
        ok, buf = cv2.imencode(
            '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY])
        if ok:
            msg.data = buf.tobytes()
            pub.publish(msg)
            published += 1

        now = time.monotonic()
        if now - stats_started >= 2.0:
            elapsed = now - stats_started
            print(f"[topdown] publish_fps={published / elapsed:.2f} (may repeat images)")
            if gen is not None:
                for td in gen["process_tds"]:
                    count = td.stat_data.frames_processed_count
                    if td.device_id in previous_processed:
                        print(f"[project {td.device_id}] fps="
                              f"{(count - previous_processed[td.device_id]) / elapsed:.2f} (includes placeholders)")
                    previous_processed[td.device_id] = count
            stats_started, published = now, 0

        remaining = frame_period - (time.monotonic() - loop_started)
        if remaining > 0:
            stop_event.wait(remaining)
    # cv2.destroyWindow("topdown")
    print("[topdown] stopped")


def main():
    rclpy.init()
    node       = Node('tmms_camera_publisher')
    stop_event = threading.Event()
    threads    = []

    # --- Lightweight 2-camera projected compositor, supervised ---
    state = {"lock": threading.Lock(), "pipeline": None}
    watchdog_t = threading.Thread(target=surround_watchdog, args=(state, stop_event), daemon=True)
    watchdog_t.start()
    threads.append(watchdog_t)

    topdown_pub = node.create_publisher(CompressedImage, '/topdown_cam/compressed', 10)
    print("Publishing /topdown_cam/compressed from front/back compositor")
    t = threading.Thread(target=topdown_thread, args=(node, topdown_pub, state, stop_event), daemon=True)
    t.start()
    threads.append(t)

    # Wrist + third-person
    for cam in CAMERAS:
        pub = node.create_publisher(CompressedImage, cam["topic"], 10)
        print(f"Publishing {cam['topic']} from {cam['device']}")
        t = threading.Thread(
            target=camera_thread,
            args=(node, pub, cam["device"], stop_event, cam.get("rotate")),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for t in threads:
            t.join()
        node.destroy_node()
        rclpy.shutdown()
        with state["lock"]:
            gen = state["pipeline"]
            state["pipeline"] = None
        if gen is not None:
            stop_pipeline(gen)


if __name__ == "__main__":
    main()
