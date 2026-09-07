import os
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from surround_view import CaptureThread, CameraProcessingThread
from surround_view import FisheyeCameraModel, BirdView
from surround_view import MultiBufferManager, ProjectedImageBuffer
import surround_view.param_settings as settings

# --- Wrist / third-person cam config ---
TARGET_HEIGHT = 480
JPEG_QUALITY  = 50

CAMERAS = [
    {"device": "/dev/wrist_cam",         "topic": "/wrist_cam/compressed"},
    {"device": "/dev/third_person_cam",  "topic": "/third_person_cam/compressed", "rotate": cv2.ROTATE_180},
]

# how often a disconnected camera retries opening its device node
RECONNECT_INTERVAL_SEC = 2.0
# how often the surround-view watchdog checks pipeline health / retries a rebuild
RECONNECT_CHECK_INTERVAL_SEC = 2.0

# --- Surround view config (must be run from the script directory) ---
yamls_dir      = os.path.join(os.getcwd(), "yaml")
camera_ids     = ["/dev/front_cam", "/dev/back_cam", "/dev/left_cam", "/dev/right_cam"]
flip_methods   = [0, 0, 2, 2]
names          = settings.camera_names
cameras_files  = [os.path.join(yamls_dir, name + ".yaml") for name in names]
camera_models  = [FisheyeCameraModel(cf, n) for cf, n in zip(cameras_files, names)]


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
    """Construct and start one fresh generation of the 4-camera surround-view
    pipeline. Returns a dict of the live objects, or None if any of the 4
    symlinks is currently missing or any camera fails to open."""
    if not all(os.path.exists(p) for p in camera_ids):
        return None

    capture_tds = [
        CaptureThread(cid, fm, resolution=(640, 480), use_gst=False)
        for cid, fm in zip(camera_ids, flip_methods)
    ]
    capture_buffer_manager = MultiBufferManager()
    for td in capture_tds:
        capture_buffer_manager.bind_thread(td, buffer_size=8)
        if not td.connect_camera():
            for t2 in capture_tds:
                t2.disconnect_camera()
            return None
        td.start()

    proc_buffer_manager = ProjectedImageBuffer()
    process_tds = [
        CameraProcessingThread(capture_buffer_manager, cid, cm)
        for cid, cm in zip(camera_ids, camera_models)
    ]
    for td in process_tds:
        proc_buffer_manager.bind_thread(td)
        td.start()

    # ProjectedImageBuffer.sync() runs before set_frame_for_device() in each
    # CameraProcessingThread's loop, so the very first completed barrier round
    # is always built from bind_thread()'s initial placeholder frames, not real
    # captured ones -- deterministically, on every pipeline start. Draining and
    # discarding that first round here (blocking, bounded by camera framerate)
    # means BirdView only ever consumes real, correctly-shaped frames.
    proc_buffer_manager.get()

    birdview = BirdView(proc_buffer_manager)
    birdview.load_weights_and_masks("./weights.png", "./masks.png")
    birdview.start()

    return dict(
        capture_tds=capture_tds,
        capture_buffer_manager=capture_buffer_manager,
        process_tds=process_tds,
        proc_buffer_manager=proc_buffer_manager,
        birdview=birdview,
    )


def stop_pipeline(gen):
    """Best-effort teardown using only the surround_view package's public API.
    All 4 CaptureThreads always stop cleanly and release their cameras. The 3
    "healthy" CameraProcessingThreads are released via wake_all(). The dead
    camera's own CameraProcessingThread and the BirdView thread are typically
    stuck forever in a blocking Buffer.get() with no public cancel/timeout --
    those are abandoned (parked, ~0% CPU, no camera handle held): a bounded,
    low-impact leak, not an unbounded/CPU-burning one.
    """
    for td in gen["capture_tds"]:
        td.stop()
    for td in gen["capture_tds"]:
        td.wait(1000)
        td.disconnect_camera()

    for td in gen["process_tds"]:
        td.stop()
    gen["proc_buffer_manager"].wake_all()
    for td in gen["process_tds"]:
        td.wait(500)

    gen["birdview"].stop()
    gen["birdview"].wait(500)


def pipeline_is_healthy(gen, last_counts):
    """Frame-count-progress check per capture thread. More reliable than
    is_camera_connected()/isOpened(), which can stay stale-True after a
    physical unplug -- frames_processed_count only increments on a genuinely
    successful grab()+add(), so it stops moving the instant a camera dies.
    A device seen for the first time gets one free grace tick."""
    healthy = True
    for td in gen["capture_tds"]:
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


def topdown_thread(node, pub, state, stop_event):
    offline_frame   = _make_offline_frame(780, 1040, "SURROUND VIEW OFFLINE")
    last_good_frame = None
    last_gen        = None
    print("[topdown] surround view publishing started")
    while not stop_event.is_set():
        with state["lock"]:
            gen = state["pipeline"]

        # a fresh generation hasn't produced anything yet -- don't reuse a
        # frame left over from a previous (possibly abandoned) generation
        if gen is not last_gen:
            last_good_frame = None
            last_gen = gen

        if gen is not None:
            # buffer.isempty() being True just means BirdView hasn't produced
            # its next frame yet (normal, happens between every production
            # cycle) -- NOT that the pipeline is down. Only .get() when a
            # frame is actually ready (guaranteed non-blocking, single
            # consumer), and keep republishing the last real frame in
            # between so this doesn't flash to the offline placeholder on
            # every empty poll.
            if not gen["birdview"].buffer.isempty():
                last_good_frame = cv2.resize(gen["birdview"].get(), (780, 1040))
            img = last_good_frame if last_good_frame is not None else offline_frame
            time.sleep(0.03)
        else:
            img = offline_frame
            time.sleep(0.05)

        msg              = CompressedImage()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.format       = "jpeg"
        _, buf           = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 20])
        msg.data         = buf.tobytes()
        pub.publish(msg)
    print("[topdown] stopped")


def main():
    rclpy.init()
    node       = Node('tmms_camera_publisher')
    stop_event = threading.Event()
    threads    = []

    # --- Surround view (4-camera stitched topdown), supervised ---
    state = {"lock": threading.Lock(), "pipeline": None}
    watchdog_t = threading.Thread(target=surround_watchdog, args=(state, stop_event), daemon=True)
    watchdog_t.start()
    threads.append(watchdog_t)

    topdown_pub = node.create_publisher(CompressedImage, '/topdown_cam/compressed', 10)
    print("Publishing /topdown_cam/compressed from surround view")
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
