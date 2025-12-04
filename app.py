import cv2
import numpy as np
import math

from models import MiniatureSettings
from processing import (
    ImageToolbox,
    render_composite,
    draw_guides_cv2,
    TiltShiftZoneDefiner,
)


class LocalMiniatureApp:
    def __init__(self, image_path: str):
        # Load and validate image
        self.raw_img = cv2.imread(image_path)
        if self.raw_img is None:
            raise ValueError("Image not found")

        # Resize if image is too large for screen
        max_h = 900
        h, w = self.raw_img.shape[:2]
        if h > max_h:
            scale = max_h / h
            self.raw_img = cv2.resize(self.raw_img, (int(w * scale), max_h))
        self.rows, self.cols = self.raw_img.shape[:2]

        # Initialize Settings with Asymmetric Defaults
        self.settings = MiniatureSettings(
            center_x=0.5,
            center_y=0.5,
            angle_upper=-90,
            dist_upper_sharp=50,
            dist_upper_trans=50,
            angle_lower=90,
            dist_lower_sharp=50,
            dist_lower_trans=50,
            saturation=1.4,
            contrast=1.1,
        )

        # Cache for performance
        self.cache_sharp = None
        self.cache_blur = None
        self.blur_strength = 50
        self.blur_shapes = [
            "gaussian",
            "circular_soft",
            "circular",
            "circular_sharp",
            "6_blades",
            "8_blades",
            "9_blades",
        ]
        self.blur_shape_idx = 0
        self.window_name = "Miniature"

        # Interaction Flags
        self.drag_center = False
        self.drag_handle_up = False
        self.drag_handle_down = False
        self.active_node = None

        # Prepare initial layers, normally we only do once this computation
        self.update_layers()

    def update_layers(self):
        """Computation of blur : please check processing.py for the function"""
        """Re-computes heavy operations only when necessary."""
        self.cache_sharp = ImageToolbox.apply_enhancements(
            self.raw_img, self.settings.saturation, self.settings.contrast
        )
        shape = self.blur_shapes[self.blur_shape_idx]
        self.cache_blur = ImageToolbox.apply_blur_style(
            self.cache_sharp, self.blur_strength, shape
        )

    def on_mouse(self, event, x, y, flags, param):
        """Mouse Events: Click, Drag, Hover."""
        """NOT VERY IMPORTANT... pure programminng"""

        # Convert Center to pixels
        cx = int(self.settings.center_x * self.cols)
        cy = int(self.settings.center_y * self.rows)

        # Get Handle positions (Handles are on the Dashed Line)
        definer = TiltShiftZoneDefiner((self.rows, self.cols), self.settings)
        h1, h2 = definer.get_handles_pos()
        HIT_SQ = 225  # Hit detection radius squared (15px)

        # --- EVENT: L-Button Down (Start Drag) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            if (x - cx) ** 2 + (y - cy) ** 2 < HIT_SQ:
                self.drag_center = True
                self.active_node = "center"
            elif (x - h1[0]) ** 2 + (y - h1[1]) ** 2 < HIT_SQ:
                self.drag_handle_up = True
                self.active_node = "up"
            elif (x - h2[0]) ** 2 + (y - h2[1]) ** 2 < HIT_SQ:
                self.drag_handle_down = True
                self.active_node = "down"

        # --- EVENT: Mouse Move (Drag or Hover) ---
        elif event == cv2.EVENT_MOUSEMOVE:
            dx, dy = x - cx, y - cy

            # 1. Dragging Center
            if self.drag_center:
                self.settings.center_x = np.clip(x / self.cols, 0, 1)
                self.settings.center_y = np.clip(y / self.rows, 0, 1)
                self.active_node = "center"

            # 2. Dragging Upper Handle
            elif self.drag_handle_up:
                # Update Angle
                angle = math.degrees(math.atan2(dy, dx))
                self.settings.angle_upper = angle

                # Update Sharp Distance (Handle follows mouse)
                dist = math.sqrt(dx**2 + dy**2)
                self.settings.dist_upper_sharp = max(10, int(dist))
                # Note: dist_upper_trans is NOT changed here. It remains fixed.

                self.active_node = "up"

            # 3. Dragging Lower Handle
            elif self.drag_handle_down:
                # Update Angle
                angle = math.degrees(math.atan2(dy, dx))
                self.settings.angle_lower = angle

                # Update Sharp Distance
                dist = math.sqrt(dx**2 + dy**2)
                self.settings.dist_lower_sharp = max(10, int(dist))

                self.active_node = "down"

            # 4. Hover Detection (When not dragging)
            else:
                self.active_node = None
                if (x - cx) ** 2 + (y - cy) ** 2 < HIT_SQ:
                    self.active_node = "center"
                elif (x - h1[0]) ** 2 + (y - h1[1]) ** 2 < HIT_SQ:
                    self.active_node = "up"
                elif (x - h2[0]) ** 2 + (y - h2[1]) ** 2 < HIT_SQ:
                    self.active_node = "down"

        # --- EVENT: L-Button Up (End Drag) ---
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_center = False
            self.drag_handle_up = False
            self.drag_handle_down = False

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # Helper for trackbars
        def nothing(x):
            pass

        # UI Controls
        cv2.createTrackbar("Blur", self.window_name, 50, 100, nothing)
        cv2.createTrackbar(
            "Kernel", self.window_name, 0, len(self.blur_shapes) - 1, nothing
        )

        print("--- Instructions ---")
        print("1. Drag Red Dot: Move Focus Center.")
        print("2. Drag Blue Dots: Rotate angle & Resize Sharp Zone.")
        print("   (Blue dots are now on the dashed line).")
        print("3. Use Sliders: Adjust the outer Fade Width independently.")
        print(
            "4. 'Kernel' slider: 0=gaussian, 1=circ_soft, 2=circ, 3=circ_sharp, 4=6 blades, 5=8 blades, 6=9 blades."
        )

        while True:
            # Sync Trackbar values
            new_blur_strength = cv2.getTrackbarPos("Blur", self.window_name)
            new_shape_idx = cv2.getTrackbarPos("Kernel", self.window_name)

            # Update cache if blur strength changed
            # This is a heavy computation and only occurs when blur_strength changes
            if (
                getattr(self, "_last_b", None) != new_blur_strength
                or getattr(self, "_last_shape_idx", None) != new_shape_idx
            ):
                self.blur_strength = new_blur_strength
                self.blur_shape_idx = new_shape_idx
                self.update_layers()

            self._last_b = self.blur_strength
            self._last_shape_idx = self.blur_shape_idx

            # 1. Composite : check processing.py
            # finsl image = composition of 2 images :
            # image clear (zone sharp) and image blurred (zone blurred), which are already computed using update_layers()
            # finally zone itermdeiate = real time interpolation of sharp and blurred
            res = render_composite(self.cache_sharp, self.cache_blur, self.settings)

            # 2. Draw Guides : check processing.py
            final = draw_guides_cv2(res, self.settings, active_node=self.active_node)

            # 3. Display
            cv2.imshow(self.window_name, final)

            # 4. Quit
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = LocalMiniatureApp("./test.jpg")
    app.run()
