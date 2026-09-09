"""Feathered front/rear display with a direct fisheye-to-display remap.

The display is a camera visualization, not a metrically stitched ground map.
No image-intensity threshold is used to decide which camera pixels are valid.
"""
import cv2
import numpy as np


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _alpha_blend(source, destination, alpha):
    """Blend BGR arrays using a float alpha map.

    This deliberately uses NumPy instead of ``cv2.blendLinear``.  The latter
    has different shape/type requirements between the OpenCV versions on the
    development machine and the dog's PC4 image.
    """
    source = np.asarray(source, dtype=np.float32)
    destination = np.asarray(destination, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    if (source.ndim != 3 or source.shape[2] != 3 or
            source.shape != destination.shape or
            alpha.shape[:2] != source.shape[:2]):
        raise ValueError("source, destination, and alpha dimensions do not match")
    return np.clip(source * alpha + destination * (1.0 - alpha), 0, 255).astype(np.uint8)


class ConeProjector:
    """Compose inverse ground projection and lens distortion before sampling.

Unlike remap(raw, 640x480) followed by warpPerspective(..., 500x225), this
does not discard rays outside the intermediate undistorted rectangle. The
original calibration matrices are used without modifying the YAML files.
"""

    def __init__(self, model, size, dog_width, peripheral_width=1.15,
                 edge_feather=28.0, near_fade=90.0, cone_half_angle=62.0):
        self.name = model.camera_name
        if self.name not in ("front", "back"):
            raise ValueError("ConeProjector supports front and back cameras")

        self.width, self.height = map(int, size)

        self.calibration_size = tuple(map(int, model.resolution))
        self.edge_feather = max(1.0, float(edge_feather))
        if peripheral_width <= 0:
            raise ValueError("peripheral_width must be positive")
        
        self.peripheral_width = max(1.0, float(peripheral_width))

        camera_matrix = np.asarray(model.camera_matrix, np.float64)
        adjusted = camera_matrix.copy()
        scale_xy = np.asarray(model.scale_xy).reshape(2)
        shift_xy = np.asarray(model.shift_xy).reshape(2)
        adjusted[0, 0] *= scale_xy[0]
        adjusted[1, 1] *= scale_xy[1]
        adjusted[0, 2] += shift_xy[0]
        adjusted[1, 2] += shift_xy[1]

        # Keep the calibrated near/far ground range. Fit its full height to
        # the panel, and widen its lateral range without cropping the image.
        # This is a display layout, not a common metric scale across views.
        project_w, project_h = model.project_shape

        self.project_w, self.project_h = project_w, project_h

        pixels_per_unit = self.width / (project_w * peripheral_width)

        yy, xx = np.indices((self.height, self.width), dtype=np.float64)

        px = (xx - (self.width - 1) / 2) / pixels_per_unit + (project_w - 1) / 2
        py = yy * (project_h - 1) / max(1, self.height - 1)

        self.px = px
        self.py = py

        inverse = np.linalg.inv(np.asarray(model.project_matrix, np.float64))
        plane = np.stack((px.ravel(), py.ravel(), np.ones(px.size)))
        undistorted = inverse @ plane
        reference = inverse @ np.array([(project_w - 1) / 2, project_h - 1, 1.0])
        denominator = undistorted[2]
        # Reject the far side of the projection horizon, where the planar
        # homography would fold unrelated rays back into the display.
        on_plane = denominator * np.sign(reference[2]) > 1e-8
        safe_denominator = np.where(on_plane, denominator, 1.0)
        undistorted /= safe_denominator
        normalized = np.linalg.inv(adjusted) @ undistorted
        normalized = np.ascontiguousarray(normalized[:2].T.reshape(-1, 1, 2))
        distorted = cv2.fisheye.distortPoints(
            normalized, camera_matrix, np.asarray(model.dist_coeffs, np.float64),
            alpha=float(camera_matrix[0, 1] / camera_matrix[0, 0]))
        self.map_x = distorted[:, 0, 0].reshape(yy.shape).astype(np.float32)
        self.map_y = distorted[:, 0, 1].reshape(yy.shape).astype(np.float32)
        self.on_plane = on_plane.reshape(yy.shape)

        distance = (self.height - 1 - yy).astype(np.float32)
        slope = np.tan(np.deg2rad(cone_half_angle))
        base_width = max(dog_width * 0.42, self.width * 0.35)
        half_width = dog_width * 0.42 + distance * slope
        # Feather perpendicular to each sloping side, not just horizontally.
        lateral = smoothstep((half_width - np.abs(xx - (self.width - 1) / 2)) /
                             (self.edge_feather * np.sqrt(1 + slope * slope)))
        # Before:
        # self.cone_alpha = (lateral * smoothstep(distance / max(1.0, near_fade))).astype(np.float32)

        # After:
        if near_fade > 0:
            near_mask = smoothstep(distance / near_fade)
        else:
            near_mask = 1.0  # Keeps immediate ground fully visible/unfeathered

        self.cone_alpha = (lateral * near_mask).astype(np.float32)
        if self.name == "back":
            # Match the legacy back-camera 180-degree flip, including its
            # left/right orientation. Rotate the map, not the finished image.
            self.map_x = np.ascontiguousarray(self.map_x[::-1, ::-1])
            self.map_y = np.ascontiguousarray(self.map_y[::-1, ::-1])
            self.on_plane = np.ascontiguousarray(self.on_plane[::-1, ::-1])
            self.cone_alpha = np.ascontiguousarray(self.cone_alpha[::-1, ::-1])
        self._input_size = None
        self._maps = None
        self._alpha = None
        self._prepare_maps(self.calibration_size)

    def _prepare_maps(self, size):
        width, height = size
        calibration_w, calibration_h = self.calibration_size
        if size != self.calibration_size:
            print(f"[view {self.name}] capture {size}, calibration {self.calibration_size}; "
                  "scaling calibration coordinates assuming the same sensor field of view")
            if abs(width / height - calibration_w / calibration_h) > 0.01:
                print(f"[view {self.name}] WARNING: aspect ratios differ; verify camera calibration/crop")
        map_x = (self.map_x + 0.5) * (width / calibration_w) - 0.5
        map_y = (self.map_y + 0.5) * (height / calibration_h) - 0.5
        valid = (self.on_plane & np.isfinite(map_x) & np.isfinite(map_y) &
                 (map_x >= 0) & (map_x < width - 1) &
                 (map_y >= 0) & (map_y < height - 1))
        # Geometric validity preserves real black/dark objects in the scene.
        # Pad before the distance transform so display borders feather too.
        # padded = np.pad(valid.astype(np.uint8), 1)
        padded = np.pad(valid.astype(np.uint8), ((2, 0), (1, 1))) # Don't pad y-bottom
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        alpha = smoothstep(distance / self.edge_feather) * self.cone_alpha
        self._alpha = np.rint(alpha * 255).astype(np.uint8)
        map_x = np.where(valid, map_x, -1).astype(np.float32)
        map_y = np.where(valid, map_y, -1).astype(np.float32)
        self._maps = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        self._input_size = size

    def project(self, image):
        """Return a BGRA layer; alpha describes geometry, never brightness."""
        if image is None or image.size == 0:
            return np.zeros((self.height, self.width, 4), np.uint8)
        size = (image.shape[1], image.shape[0])
        if size != self._input_size:
            self._prepare_maps(size)
        raw = np.ascontiguousarray(image[:, :, :3])
        bgr = cv2.remap(raw, *self._maps, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        layer = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        layer[:, :, 3] = self._alpha
        return layer


class TwoCameraView:
    """Cached cone geometry and a softly blended robot on a charcoal canvas."""

    # def __init__(self, models, dog_image, width=780, height=1040,
    #              dog_height=150, peripheral_width=1.15, edge_feather=28,
    #              near_fade=0, cone_half_angle=62, show_labels=True):
    def __init__(self, models, dog_image, width=780, height=1040,
                 dog_height=150, peripheral_width=1.15, edge_feather=28,
                 near_fade=0, cone_half_angle=62, show_labels=True):
        self.width, self.height = int(width), int(height)
        self.show_labels = show_labels
        self.dog = None
        if dog_image is not None and dog_image.size:
            dog_height = min(int(dog_height), self.height // 3)
            dog_width = max(1, round(dog_image.shape[1] * dog_height / dog_image.shape[0]))
            if dog_width > self.width // 3:
                dog_height = max(1, round(dog_height * (self.width // 3) / dog_width))
                dog_width = self.width // 3
            self.dog = cv2.resize(dog_image, (dog_width, dog_height), interpolation=cv2.INTER_AREA)
        else:
            dog_height, dog_width = min(int(dog_height), self.height // 3), 70
        self.dog_rect = ((self.width - dog_width) // 2dog, (self.height - dog_height) // 2,
                         dog_width, dog_height)
        _, dog_y, _, dog_h = self.dog_rect
        # The cone tips extend slightly underneath the icon, removing the
        # rectangular dead space between the panels and the robot.
        # overlap = max(2, round(dog_h * 0.08))
        # overlap = 0
        overlap = max(4, round(dog_h * 0.12))
        front_h = dog_y + overlap
        rear_y = dog_y + dog_h - overlap
        
        self.regions = {"front": (0, front_h), "back": (rear_y, self.height - rear_y)}
        self.projectors = {}
        for model in models:
            _, panel_h = self.regions[model.camera_name]
            self.projectors[model.camera_name] = ConeProjector(
                model, (self.width, panel_h), dog_width, peripheral_width,
                edge_feather, near_fade, cone_half_angle)

        project_w, project_h = models[0].project_shape
        self.px = self.width / (project_w * peripheral_width)
        self.py = self.height / project_h

        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        halo = np.exp(-((xx - self.width / 2) / (dog_width * 1.25)) ** 2 -
                      ((yy - self.height / 2) / (dog_h * 0.9)) ** 2)
        self.background = np.empty((self.height, self.width, 3), np.uint8)
        self.background[:] = (20, 22, 25)
        self.background = np.clip(self.background.astype(np.float32) + halo[:, :, None] * 5,
                                  0, 255).astype(np.uint8)
        if self.dog is not None:
            if self.dog.shape[2] == 4:
                self.dog_alpha = self.dog[:, :, 3].astype(np.float32) / 255
            else:
                # The supplied dog PNG is RGB, not transparent. Feather only
                # its rectangle's border; never erase dark parts of the robot.
                mask = np.pad(np.ones(self.dog.shape[:2], np.uint8), 1)
                edge = cv2.distanceTransform(mask, cv2.DIST_L2, 3)[1:-1, 1:-1]
                self.dog_alpha = smoothstep(edge / max(3.0, dog_width * 0.06))

    """ This function is required for _draw_distance_labels to work properly """
    def _draw_arc_label(self, canvas, text, center, radius_x, radius_y, angle_deg, color, y_offset=0):
        angle = np.deg2rad(angle_deg)

        # Calculating the position of the label to be placed at the ends of the arc
        x = int(center[0] + radius_x * np.cos(angle))
        y = int(center[1] + radius_y * np.sin(angle))

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )

        padding = 6

        if angle_deg in (0, 360):
            origin = (x + padding, y - 6 + y_offset)
        elif angle_deg == 180:
            origin = (x - tw - padding, y - 6 + y_offset)

        # White outline improves readability over camera footage
        cv2.putText(canvas, text, origin, font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, font, scale,
                    color, thickness, cv2.LINE_AA)

    def _draw_distance_labels(self, canvas):
        front_center_x = (self.width // 2)
        front_center_y = (self.height // 2) - 65

        back_center_y = (self.height // 2) + 65

        # pixels_per_cm_x = self.width / (self.project_w * self.peripheral_width)  # these are defined in ConeProjector init
        # pixels_per_cm_y = self.height / self.project_h

        front_distances = [
            # (10, (0, 0, 255)),      # red, BGR
            ("20", 19, (180, 105, 255)),    # pink ACTUAL LINE IS AT: 20CM
            ("30", 30, (0, 255, 255)),    # yellow ACTUAL IS AT: 30CM
            ("50", 55, (0, 255, 0)),      # green ACTUAL IS AT: 50CM
            # (100, (0, 255, 0)),
            # (200, (0, 100, 100)),
        ]

        back_distances =[
            (50, (0, 255, 0)),
        ]

        for distance_text, distance_cm, color in front_distances:
            radius_x = round(distance_cm * self.px * 0.8)
            radius_y = round(distance_cm * self.py * 0.3)

            center = (front_center_x, front_center_y)

            label = f"{distance_text}"

            if distance_text == "20":
                self._draw_arc_label(
                    canvas, label, center,
                    radius_x, radius_y,
                    180, color, y_offset=20
                )

                self._draw_arc_label(
                    canvas, label, center,
                    radius_x, radius_y,
                    360, color, y_offset=20
                )
                
            else:
                self._draw_arc_label(
                    canvas, label, center,
                    radius_x, radius_y,
                    180, color
                )

                self._draw_arc_label(
                    canvas, label, center,
                    radius_x, radius_y,
                    360, color
                )
            
        for distance_cm, color in back_distances:
            radius_x = round(distance_cm * self.px * 0.8)
            radius_y = round(distance_cm * self.py * 0.3)

            center = (front_center_x, back_center_y)

            label = f"{distance_cm} CM"

            self._draw_arc_label(
                canvas, label, center,
                radius_x, radius_y,
                0, color
            )

            self._draw_arc_label(
                canvas, label, center,
                radius_x, radius_y,
                180, color
            )

    def _shade_distance_band(self, canvas, center):
        inner_axes = (
            round(19 * self.px * 0.8),   # calibrated 20 cm arc
            round(19 * self.py * 0.3),
        )

        outer_axes = (
            round(30 * self.px * 0.8),   # calibrated 30 cm arc
            round(30 * self.py * 0.3),
        )

        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)

        # Fill the area inside the 30 cm arc.
        cv2.ellipse(
            mask, center, outer_axes,
            0, 180, 360, 255, -1
        )

        # Remove the area inside the 20 cm arc.
        cv2.ellipse(
            mask, center, inner_axes,
            0, 180, 360, 0, -1
        )

        shade = np.full_like(canvas, (80, 220, 255))  # light yellow, BGR
        blended = cv2.addWeighted(canvas, 0.78, shade, 0.22, 0)

        canvas[mask > 0] = blended[mask > 0]

    def _draw_distance_arcs(self, canvas):
        """Draw concentric distance arcs on the front camera panel."""
        front_center_x = (self.width // 2)
        front_center_y = (self.height // 2) - 65

        back_center_y = (self.height // 2) + 65

        front_center = (front_center_x, front_center_y)

        # Shade between the calibrated 20 cm and 30 cm arcs.
        self._shade_distance_band(canvas, front_center)

        # pixels_per_cm_x = self.width / (self.project_w * self.peripheral_width)  # these are defined in ConeProjector init
        # pixels_per_cm_y = self.height / self.project_h

        front_distances = [
            (19, (180, 105, 255)),    # pink ACTUAL ARC IS AT: 20CM , BGR format
            (30, (0, 255, 255)),      # yellow ACTUAL ARC IS AT: 30CM , BGR format
            (55, (0, 255, 0)),        # green ACTUAL ARC IS AT: 50CM , BGR format
        ]

        back_distances =[
            (50, (0, 255, 0)),        # green ACTUAL ARC IS AT: 50CM , BGR format
        ]

        for distance_cm, color in front_distances:
            radius_x = round(distance_cm * self.px * 0.8)
            radius_y = round(distance_cm * self.py * 0.3)

            center = (front_center_x, front_center_y)

            cv2.ellipse(
                canvas,
                (front_center_x, front_center_y),
                (radius_x, radius_y),
                0,
                190,
                350,
                color,
                1,
                cv2.LINE_AA,
            )
            
        for distance_cm, color in back_distances:
            radius_x = round(distance_cm * self.px * 0.8)
            radius_y = round(distance_cm * self.py * 0.3)

            center = (front_center_x, back_center_y)

            cv2.ellipse(
                canvas,
                (front_center_x, back_center_y),
                (radius_x, radius_y),
                0,
                10,
                170,
                color,
                1,
                cv2.LINE_AA,
            )

    def compose(self, frames):
        """frames maps 'front'/'back' to already projected BGRA layers."""
        canvas = self.background.copy()
        for name, (y, panel_h) in self.regions.items():
            layer = frames.get(name)
            # The legacy buffer initially contains differently sized black
            # placeholders; wait for the new projector's correctly sized layer.
            if layer is None or layer.shape != (panel_h, self.width, 4):
                continue
            bgr = layer[:, :, :3]
            alpha = layer[:, :, 3].astype(np.float32) / 255.0
            canvas[y:y + panel_h] = _alpha_blend(
                bgr, canvas[y:y + panel_h], alpha)

        self._draw_distance_arcs(canvas)

        if self.dog is not None:
            x, y, w, h = self.dog_rect
            canvas[y:y + h, x:x + w] = _alpha_blend(
                self.dog[:, :, :3], canvas[y:y + h, x:x + w], self.dog_alpha)

        self._draw_distance_labels(canvas)        

        if self.show_labels:
            for label, y in (("FRONT", 32), ("REAR", self.height - 20)):
                cv2.putText(canvas, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (174, 181, 189), 1, cv2.LINE_AA)
        return canvas
