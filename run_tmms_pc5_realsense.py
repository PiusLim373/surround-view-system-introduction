#!/usr/bin/env python3
"""pc5 camera publisher.

Opens the two RealSense colour nodes pinned by udev (/dev/wrist_cam and
/dev/third_person_cam) and republishes each as a JPEG CompressedImage topic.
The third-person camera is mounted upside down, so its frames are rotated 180.

No surround view / fisheye projection here -- that lives in run_tmms.py on pc4.
"""

import os
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# Capture settings applied to every camera below.
CAPTURE_FOURCC = "MJPG"
CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS    = 15

# Published frames are resized to this height (aspect ratio preserved).
TARGET_HEIGHT = 480
JPEG_QUALITY  = 50

# how often a disconnected camera retries opening its device node
RECONNECT_INTERVAL_SEC = 2.0

CAMERAS = [
    {"device": "/dev/wrist_cam",         "topic": "/wrist_cam/compressed"},
    {"device": "/dev/third_person_cam",  "topic": "/third_person_cam/compressed",
     "rotate": cv2.ROTATE_180},
]


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
    """Publish one camera forever, tolerating an absent or unplugged device.

    While the device node is missing or unreadable the topic keeps producing an
    OFFLINE placeholder, and the open is retried every RECONNECT_INTERVAL_SEC,
    so a replug recovers without restarting the script.
    """
    label = device_path.rsplit("/", 1)[-1]
    offline_frame = _make_offline_frame(CAPTURE_WIDTH, TARGET_HEIGHT, f"{label} OFFLINE")

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
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAPTURE_FOURCC))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
                        connected = True
                        # Report the mode actually negotiated -- the driver may
                        # silently fall back (e.g. to YUYV) if MJPG is refused.
                        w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps        = cap.get(cv2.CAP_PROP_FPS)
                        raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        fourcc_str = "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4))
                        print(f"[{label}] opened: {w}x{h} @ {fps}fps  fourcc={fourcc_str} "
                              f"({os.path.realpath(device_path)})")
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


def main():
    rclpy.init()
    node       = Node('tmms_camera_publisher')
    stop_event = threading.Event()
    threads    = []

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


if __name__ == "__main__":
    main()
