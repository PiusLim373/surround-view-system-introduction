# TMMS two-camera runner

This folder contains the runtime for `run_tmms_2cams.py`. The program opens the
front and back fisheye cameras, projects them using their calibration files,
composes a two-camera view, and publishes it as a ROS 2 compressed-image topic.

## What the program does

`run_tmms_2cams.py`:

- Opens `/dev/front_cam` and `/dev/back_cam`.
- Reads each camera's capture mode from `camera_config.json`.
- Applies the calibration and projection matrices from `yaml/front.yaml` and
  `yaml/back.yaml`.
- Composes the projected images around `images/dog.png`.
- Publishes the result on `/topdown_cam/compressed`.
- Publishes an offline/placeholder image while a camera is unavailable and
  retries the pipeline when a camera stalls.

The front and back streams are processed independently; they are not hardware
frame-synchronised.

## Important files

### Runtime code

| File | Purpose |
| --- | --- |
| `run_tmms_2cams.py` | Main executable. Camera selection, output size, display tuning, ROS topic, watchdog, and capture settings are defined here. |
| `camera_config_loader.py` | Loads camera capture metadata from JSON. It supports stable `/dev/v4l/by-path/` identities and fallback `/dev/videoN` nodes. |
| `surround_view/` | Camera capture, buffering, fisheye projection, image processing, and view-composition library used by the runner. |
| `surround_view/two_camera_view.py` | Composes the front/back projected layers and the center image. |
| `discover_cameras.py` | Detects cameras and generates a fresh `camera_config.json` from their advertised V4L2 modes. |
| `pius_test.py` | Optional camera/USB-bandwidth diagnostic tool. Use it before deployment when both cameras do not stream reliably. |

### Hardware and configuration

| File or directory | Purpose |
| --- | --- |
| `99-tmms-cams.rules` | udev rules intended to create `/dev/front_cam` and `/dev/back_cam` (plus optional left/right and wrist/third-person aliases). |
| `camera_config.json` | Active per-camera capture metadata: stable path, pixel format, width, height, and FPS. This is the file read by the runner. |
| `camera_config.*.json` | Saved alternative capture profiles. They are not used automatically; copy one to `camera_config.json` only after checking that it matches the current hardware. |
| `yaml/front.yaml` | Front camera intrinsic matrix, distortion coefficients, calibration resolution, and projection matrix. |
| `yaml/back.yaml` | Back camera equivalent of `front.yaml`. |
| `images/dog.png` | Center/robot image placed in the composed view. |
| `images/` | Other sample images; they are not required by the current two-camera runner. |

### Reference and diagnostic material

`doc/two_camera_view_tuning.md` describes view tuning. The other files under
`doc/` record calibration, USB, camera, and troubleshooting notes. The files
`both-*.json` and `usb*-alone.json` are diagnostic reports, not runtime
configuration files.

## Things that can be changed

### Usually safe to change in `run_tmms_2cams.py`

Near the top of the file:

- `OUTPUT_WIDTH`, `OUTPUT_HEIGHT`: published composed-image dimensions.
- `OUTPUT_FPS`: target publish rate.
- `OUTPUT_JPEG_QUALITY`: JPEG quality for the composed ROS image.
- `DOG_DISPLAY_HEIGHT`: size of the center dog/robot image.
- `VIEW_PERIPHERAL_WIDTH`: widen or narrow the projected peripheral view.
- `VIEW_EDGE_FEATHER`: soften the camera-coverage boundary.
- `VIEW_NEAR_FADE`: fade projected pixels near the center; `0` keeps them visible.
- `VIEW_CONE_HALF_ANGLE`: increase to expose more side coverage.
- `VIEW_SHOW_LABELS`: show or hide `front`/`back` labels.
- `RECONNECT_INTERVAL_SEC` and `RECONNECT_CHECK_INTERVAL_SEC`: retry timing.

The optional `CAMERAS` list can be populated to publish additional direct
camera streams, for example a wrist camera. Each entry needs a device path and
ROS topic; `rotate: cv2.ROTATE_180` can rotate a camera image.

### Change only when hardware/calibration changes

- `camera_ids`: must match the udev aliases used for the physical front and
  back cameras.
- `names`: must stay in the same order as `camera_ids` and the calibration YAMLs.
- `cameras_files`: points to the calibration files used by each camera.
- `POSITION_TO_CONFIG_KEY`: must point to the stable by-path entries for the
  same physical cameras as `front_cam` and `back_cam`.
- `DEFAULT_CAPTURE_SETTINGS`: fallback mode when a position is absent from the
  JSON. It must be supported by the camera and should match calibration.
- `yaml/front.yaml` and `yaml/back.yaml`: do not edit matrix values manually
  unless the cameras have been recalibrated.

The capture resolution should match the `resolution` stored in the relevant
calibration YAML. The runner prints a warning when they differ, but projection
quality can be incorrect.

### Change in `camera_config.json`

For each active stable-path entry, the meaningful capture fields are:

```json
{
  "fourcc": "MJPG",
  "width": 640,
  "height": 480,
  "fps": 30.0
}
```

Use only modes reported by the camera. The preferred way to update this file is
to reconnect the cameras and run `discover_cameras.py`; then verify that the
front/back stable paths still correspond to the aliases in the udev setup.

## One-time setup

The following assumes a Linux machine with ROS 2 and V4L2 cameras:

1. Install the required system/runtime packages for the ROS 2 distribution,
   including `rclpy`, `sensor_msgs`, OpenCV, NumPy, and `v4l2-ctl`.
2. Source ROS 2 in every shell used to run the program:

   ```bash
   source /opt/ros/<your_ros2_distribution>/setup.bash
   ```

3. Install the udev rule, after checking that its `ID_PATH` values match the
   actual machine. The values can be inspected with `udevadm info` or
   `v4l2-ctl --list-devices`:

   ```bash
   sudo cp 99-tmms-cams.rules /etc/udev/rules.d/99-tmms-cams.rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger --subsystem-match=video4linux
   udevadm settle
   ```

4. Confirm the aliases and stable paths:

   ```bash
   ls -l /dev/front_cam /dev/back_cam
   ls -l /dev/v4l/by-path/
   v4l2-ctl --list-devices
   ```

5. If the cameras or USB ports changed, regenerate the capture configuration:

   ```bash
   python3 discover_cameras.py --out camera_config.json
   ```

6. Edit `POSITION_TO_CONFIG_KEY` in `run_tmms_2cams.py` so its `front` and
   `back` paths identify the same physical cameras as `/dev/front_cam` and
   `/dev/back_cam`. This mapping is intentionally checked at startup.

## Running `run_tmms_2cams.py`

From this directory:

```bash
cd /path/to/run_tmms_ws
source /opt/ros/<your_ros2_distribution>/setup.bash
python3 -u run_tmms_2cams.py
```

Expected output includes a successful front/back pipeline message and:

```text
Publishing /topdown_cam/compressed from front/back compositor
```

In another sourced ROS 2 shell, verify the topic:

```bash
ros2 topic list | grep topdown
ros2 topic hz /topdown_cam/compressed
```

Stop with `Ctrl+C`. The runner stops worker threads and releases the camera
devices during shutdown.

## Troubleshooting checklist

- **`/dev/front_cam` or `/dev/back_cam` is missing:** reload/fix the udev rule,
  reconnect the cameras, and inspect `udevadm` output.
- **Alias points to the wrong camera:** correct the rule or physical-port
  mapping. The runner checks aliases against `POSITION_TO_CONFIG_KEY`.
- **`no camera_config.json entry`:** regenerate the JSON or correct its stable
  path key; the runner will use the fallback mode, but this should be treated
  as a configuration issue.
- **Calibration mismatch warning:** make the JSON capture resolution match the
  corresponding YAML calibration resolution, or recalibrate the camera.
- **Black/placeholder images or low real FPS:** test each camera alone and
  together with `pius_test.py`; reduce resolution/FPS or use an appropriate
  saved `camera_config.*.json` profile if USB bandwidth is insufficient.
- **Wrong view orientation:** change the corresponding value in `flip_methods`
  or recalibrate/recreate the affected YAML.
