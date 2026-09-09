import threading
import time

import cv2
import numpy as np

from .base_thread import BaseThread
from .structures import ImageFrame
from .utils import gstreamer_pipeline

OFFLINE_TIMEOUT_SEC = 1.0
RETRY_INTERVAL_SEC = 0.01
RECONNECT_TIMEOUT_SEC = 5.0


class CaptureThread(BaseThread):
    def __init__(self,
                 device_id,
                 flip_method=0,
                 drop_if_full=True,
                 api_preference=cv2.CAP_GSTREAMER,
                 resolution=(640, 480),
                 use_gst=True,
                 fourcc="MJPG",
                 frps=None,
                 parent=None,
                 reconnect=True):
        
        super().__init__(parent)
        self.device_id = device_id
        self.flip_method = flip_method
        self.use_gst = use_gst
        self.drop_if_full = drop_if_full
        self.api_preference = api_preference
        self.resolution = resolution
        self.fourcc = fourcc
        self.frps = frps  # BaseThread.fps is a statistics Queue, NOT a frame rate.
        self.reconnect = reconnect
        self.cap = cv2.VideoCapture()
        self.buffer_manager = None
        self.last_real_frame_time = None
        self.is_offline = False
        self._stats_lock = threading.Lock()
        self._stats = dict(real_frames=0,
                           placeholder_frames=0,
                           grab_failures=0,
                           retrieve_failures=0,
                           reconnect_attempts=0,
                           grab_seconds=0.0,
                           retrieve_seconds=0.0,
                           last_error=None,
                           opened=False)
        self.reported_mode = {}
        self.setting_results = {}

    def diagnostics(self):
        """Snapshot without accessing VideoCapture from a second thread."""
        with self._stats_lock:
            result = dict(self._stats)
            result["sample_time"] = time.monotonic()
            result["last_real_age_sec"] = (
                None if self.last_real_frame_time is None else
                result["sample_time"] - self.last_real_frame_time)
            result["reported_mode"] = dict(self.reported_mode)
            result["setting_results"] = dict(self.setting_results)
        return result

    def _capture_frame(self):
        """Count only successfully grabbed AND decoded, nonempty images."""
        started = time.monotonic()
        grabbed = False
        decoded = False
        frame = None
        error = None
        grabbed_at = started
        try:
            grabbed = self.cap.grab()
            grabbed_at = time.monotonic()
            if grabbed:
                decoded, frame = self.cap.retrieve()
                decoded = decoded and frame is not None and frame.size > 0
        except cv2.error as exc:
            error = str(exc)
        finished = time.monotonic()
        if not grabbed:
            grabbed_at = finished
        with self._stats_lock:
            self._stats["grab_seconds"] += grabbed_at - started
            self._stats["retrieve_seconds"] += finished - grabbed_at
            if not grabbed:
                self._stats["grab_failures"] += 1
                self._stats["last_error"] = error or "grab() returned False"
            elif not decoded:
                self._stats["retrieve_failures"] += 1
                self._stats["last_error"] = error or "retrieve() returned an invalid image"
            else:
                self._stats["real_frames"] += 1
                self.last_real_frame_time = finished
        return frame if decoded else None

    def run(self):
        if self.buffer_manager is None:
            raise ValueError("Capture thread must be bound to a buffer manager")
        started = time.monotonic()
        last_reconnect = started
        last_placeholder = 0.0
        last_error_log = started - 2.0
        fps_started = started
        fps_count = 0
        placeholder_period = 1.0 / max(30.0, self.frps or 30.0)
        width, height = self.resolution
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            while True:
                self.stop_mutex.lock()
                stopped = self.stopped
                self.stop_mutex.unlock()
                if stopped:
                    break
                # USB cameras in pius_test/run_tmms are bound with sync=False.
                self.buffer_manager.sync(self.device_id)
                frame = self._capture_frame()
                now = time.monotonic()
                is_placeholder = frame is None
                if frame is None:
                    age = now - (self.last_real_frame_time or started)
                    if now - last_error_log >= 2.0:
                        print(f"[cam {self.device_id}] {self.diagnostics()['last_error']}")
                        last_error_log = now
                    if age >= OFFLINE_TIMEOUT_SEC:
                        if not self.is_offline:
                            print(f"[cam {self.device_id}] OFFLINE - using black placeholder")
                        self.is_offline = True
                        if now - last_placeholder >= placeholder_period:
                            frame = black_frame
                            last_placeholder = now
                            with self._stats_lock:
                                self._stats["placeholder_frames"] += 1
                    if (self.reconnect and age >= RECONNECT_TIMEOUT_SEC and
                            now - last_reconnect >= RECONNECT_TIMEOUT_SEC):
                        with self._stats_lock:
                            self._stats["reconnect_attempts"] += 1
                        print(f"[cam {self.device_id}] reopening; waiting for real frames")
                        self.disconnect_camera()
                        self.connect_camera()
                        last_reconnect = time.monotonic()
                        # Reopening does not reset the last REAL frame timestamp.
                else:
                    if self.is_offline:
                        print(f"[cam {self.device_id}] real frames resumed")
                    self.is_offline = False
                    fps_count += 1

                if now - fps_started >= 1.0:
                    self.stat_data.average_fps = fps_count / (now - fps_started)
                    fps_started, fps_count = now, 0
                if frame is not None:
                    img_frame = ImageFrame(int(time.time() * 1000) % 86400000, frame)
                    self.buffer_manager.get_device(self.device_id).add(
                        img_frame, self.drop_if_full)
                    # Legacy watchdog counter is buffer-feed progress, including
                    # placeholders. Use diagnostics()['real_frames'] for FPS.
                    self.stat_data.frames_processed_count += 1
                    self.update_statistics_gui.emit(self.stat_data)
                if is_placeholder:
                    time.sleep(RETRY_INTERVAL_SEC)
        finally:
            # Only the capture worker releases the handle while it is running.
            self.disconnect_camera()

    def connect_camera(self):
        if self.use_gst:
            options = gstreamer_pipeline(cam_id=self.device_id, flip_method=self.flip_method)
            self.cap.open(options, self.api_preference)
        else:
            self.cap.open(self.device_id, cv2.CAP_V4L2)
        with self._stats_lock:
            self._stats["opened"] = self.cap.isOpened()
        if not self.cap.isOpened():
            print(f"[cam {self.device_id}] cannot open device")
            return False

        # Configure the SAME handle that streams; a v4l2-ctl subprocess can
        # configure a different handle and its settings may be reset on open.
        # Select compression before requesting a high resolution/frame rate.
        properties = [("fourcc", cv2.CAP_PROP_FOURCC,
                       cv2.VideoWriter_fourcc(*self.fourcc))]
        if self.resolution is not None:
            properties.extend([("width", cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0]),
                               ("height", cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])])
        if self.frps is not None:
            properties.append(("fps", cv2.CAP_PROP_FPS, self.frps))
        if not self.use_gst:
            properties.append(("buffersize", cv2.CAP_PROP_BUFFERSIZE, 2))
        results = {}
        for name, prop, value in properties:
            results[name] = bool(self.cap.set(prop, value))
            if not results[name]:
                print(f"[cam {self.device_id}] WARNING: set({name}={value}) rejected")
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        mode = dict(width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=self.cap.get(cv2.CAP_PROP_FPS),
                    fourcc="".join(chr((code >> (8 * i)) & 255) for i in range(4)))
        if self.resolution is None:
            self.resolution = (mode["width"], mode["height"])
        with self._stats_lock:
            self.setting_results = results
            self.reported_mode = mode
            self._stats["opened"] = self.cap.isOpened()
        print(f"[cam {self.device_id}] requested {self.fourcc} {self.resolution} "
              f"@ {self.frps} fps; driver reports {mode} (measure real FPS below)")
        if (mode["fourcc"] != self.fourcc or
                (mode["width"], mode["height"]) != tuple(self.resolution) or
                (self.frps is not None and abs(mode["fps"] - self.frps) > 0.1)):
            print(f"[cam {self.device_id}] WARNING: driver mode differs from request")
        return self.cap.isOpened()

    def disconnect_camera(self):
        was_open = self.cap.isOpened()
        self.cap.release()
        with self._stats_lock:
            self._stats["opened"] = False
        return was_open

    def is_camera_connected(self):
        return self.diagnostics()["opened"]
