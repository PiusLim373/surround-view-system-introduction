import os
import cv2


camera_names = ["front", "back", "left", "right"]

# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 90
shift_h = 75

# size of the gap between the calibration pattern and the dog image
# in horizontal and vertical directions
inn_shift_w = 55
inn_shift_h = 70

# total width/height of the stitched image
total_w = 320 + 2 * shift_w
total_h = 400 + 2 * shift_h

# four corners of the rectangular region occupied by the dog image
# top-left (x_left, y_top), bottom-right (x_right, y_bottom)
xl = shift_w + 80 + inn_shift_w
xr = total_w - xl
yt = shift_h + 80 + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------

project_shapes = {
    "front": (total_w, yt),
    "back":  (total_w, yt),
    "left":  (total_h, xl),
    "right": (total_h, xl)
}

# pixel locations of the four points to be chosen.
# you must click these pixels in the same order when running
# the get_projection_map.py script
project_keypoints = {
    "front": [(shift_w + 80, shift_h),
              (shift_w + 80, shift_h + 80),
              (total_w - shift_w - 80, shift_h + 80),
              (total_w - shift_w - 80, shift_h)],

    "back":  [(shift_w + 80, shift_h),
              (shift_w + 80, shift_h + 80),
              (total_w - shift_w - 80, shift_h + 80),
              (total_w - shift_w - 80, shift_h)],

    "left":  [(shift_h + 80, shift_w),
              (shift_h + 80, shift_w + 80),
              (total_h - shift_h - 80, shift_w + 80),
              (total_h - shift_h - 80, shift_w)],

    "right": [(shift_h + 80, shift_w),
              (shift_h + 80, shift_w + 80),
              (total_h - shift_h - 80, shift_w + 80),
              (total_h - shift_h - 80, shift_w)],
}

dog_image = cv2.imread(os.path.join(os.path.dirname(__file__), "..", "images", "dog.png"))
dog_image = cv2.resize(dog_image, (xr - xl, yb - yt))
