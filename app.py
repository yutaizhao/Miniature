import cv2
import numpy as np
import math

from models import MiniatureSettings
from processing import ImageToolbox, render_composite, render_composite_adv, draw_guides_cv2, TiltShiftZoneDefiner

class LocalMiniatureApp:
    def __init__(self, image_path: str):
        # Load and validate image
        self.raw_img = cv2.imread(image_path)
        if self.raw_img is None: raise ValueError("Image not found")
        
        # Resize if image is too large for screen
        max_h = 900
        h, w = self.raw_img.shape[:2]
        if h > max_h:
            scale = max_h / h
            self.raw_img = cv2.resize(self.raw_img, (int(w*scale), max_h))
        self.rows, self.cols = self.raw_img.shape[:2]
        
        # Initialize Settings with Asymmetric Defaults
        self.settings = MiniatureSettings(
            center_x=0.5, center_y=0.5,
            angle_upper=-90, dist_upper_sharp=50, dist_upper_trans=50,
            angle_lower=90, dist_lower_sharp=50, dist_lower_trans=50,
            saturation=1.4, contrast=1.1
        )
        
        # 3 Images !!!
        self.cache_sharp = None
        self.cache_blur_up = None   # Blur layer for the Upper side
        self.cache_blur_low = None  # Blur layer for the Lower side
        
        self.blur_str_up = 50
        self.blur_str_low = 50
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
        # 1. Generate Sharp Base
        self.cache_sharp = ImageToolbox.apply_enhancements(
            self.raw_img, self.settings.saturation, self.settings.contrast
        )
        # 2. Generate Upper Blur
        self.cache_blur_up = ImageToolbox.apply_blur_style(
            self.cache_sharp, self.blur_str_up, "gaussian"
        )
        # 3. Generate Lower Blur
        self.cache_blur_low = ImageToolbox.apply_blur_style(
            self.cache_sharp, self.blur_str_low, "gaussian"
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
        HIT_SQ = 225 # Hit detection radius squared (15px)

        # --- EVENT: L-Button Down (Start Drag) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            if (x-cx)**2 + (y-cy)**2 < HIT_SQ:
                self.drag_center = True
                self.active_node = "center"
            elif (x-h1[0])**2 + (y-h1[1])**2 < HIT_SQ:
                self.drag_handle_up = True
                self.active_node = "up"
            elif (x-h2[0])**2 + (y-h2[1])**2 < HIT_SQ:
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
                if (x-cx)**2 + (y-cy)**2 < HIT_SQ: self.active_node = "center"
                elif (x-h1[0])**2 + (y-h1[1])**2 < HIT_SQ: self.active_node = "up"
                elif (x-h2[0])**2 + (y-h2[1])**2 < HIT_SQ: self.active_node = "down"

        # --- EVENT: L-Button Up (End Drag) ---
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_center = False
            self.drag_handle_up = False
            self.drag_handle_down = False

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        # Helper for trackbars
        def nothing(x): pass
        
        # UI Controls
        cv2.createTrackbar("Blur UP", self.window_name, 50, 100, nothing)
        cv2.createTrackbar("Blur LOW", self.window_name, 50, 100, nothing)

        print("--- Instructions ---")
        print("1. Drag Red Dot: Move Focus Center.")
        print("2. Drag Blue Dots: Rotate angle & Resize Sharp Zone.")
        print("   (Blue dots are now on the dashed line).")
        print("3. Use Sliders: Adjust the outer Fade Width independently.")

        while True:
            # Sync Trackbar values
            new_blur_up = cv2.getTrackbarPos("Blur UP", self.window_name)
            new_blur_low = cv2.getTrackbarPos("Blur LOW", self.window_name)
            
            # Check if we need to re-calculate heavy layers
            # Only update if the specific blur strength changed
            need_update = False
            if new_blur_up != self.blur_str_up:
                self.blur_str_up = new_blur_up
                need_update = True
            if new_blur_low != self.blur_str_low:
                self.blur_str_low = new_blur_low
                need_update = True
            
            if need_update:
                self.update_layers()

            # Composite : check processing.py
            # finsl image = composition of 3 images :
            # image clear (zone sharp) and 2 images blurred (up/low zones blurred),
            # which are already computed using update_layers()
            # finally zone itermdeiate = real time interpolation of sharp and blurred
            res = render_composite_adv(
                self.cache_sharp,
                self.cache_blur_up,
                self.cache_blur_low,
                self.settings
            )
            
            # 2. Draw Guides : check processing.py
            final = draw_guides_cv2(res, self.settings, active_node=self.active_node)
            
            # 3. Display
            cv2.imshow(self.window_name, final)
            
            # 4. Quit
            if cv2.waitKey(1) & 0xFF == 27: break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LocalMiniatureApp("./test.jpg")
    app.run()
