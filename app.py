import cv2
import numpy as np
import math

from models import MiniatureSettings
from processing import ImageToolbox, render_composite, draw_guides_cv2, TiltShiftZoneDefiner

class LocalMiniatureApp:
    def __init__(self, image_path: str):
        self.raw_img = cv2.imread(image_path)
        if self.raw_img is None: raise ValueError("Image not found")
        
        # Resize for display
        max_h = 900
        h, w = self.raw_img.shape[:2]
        if h > max_h:
            scale = max_h / h
            self.raw_img = cv2.resize(self.raw_img, (int(w*scale), max_h))
        self.rows, self.cols = self.raw_img.shape[:2]
        
        # Init Settings (Asymmetric defaults)
        self.settings = MiniatureSettings(
            center_x=0.5, center_y=0.5, angle_degree=0,
            upper_sharp=100, upper_trans=150,
            lower_sharp=100, lower_trans=150,
            saturation=1.4, contrast=1.1
        )
        
        self.cache_sharp = None
        self.cache_blur = None
        self.blur_strength = 50
        self.window_name = "Asymmetric Miniature Effect"
        
        # Interaction Flags
        self.drag_center = False
        self.drag_handle_up = False
        self.drag_handle_down = False
        
        self.update_layers()

    def update_layers(self):
        self.cache_sharp = ImageToolbox.apply_enhancements(
            self.raw_img, self.settings.saturation, self.settings.contrast
        )
        self.cache_blur = ImageToolbox.apply_blur_style(
            self.cache_sharp, self.blur_strength, "gaussian"
        )

    def on_mouse(self, event, x, y, flags, param):
        cx = int(self.settings.center_x * self.cols)
        cy = int(self.settings.center_y * self.rows)
        
        definer = TiltShiftZoneDefiner((self.rows, self.cols), self.settings)
        h1, h2 = definer.get_handles_pos() # [Upper, Lower]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Center
            if (x-cx)**2 + (y-cy)**2 < 200: self.drag_center = True
            # Check Handle 1 (Upper)
            elif (x-h1[0])**2 + (y-h1[1])**2 < 200: self.drag_handle_up = True
            # Check Handle 2 (Lower)
            elif (x-h2[0])**2 + (y-h2[1])**2 < 200: self.drag_handle_down = True

        elif event == cv2.EVENT_MOUSEMOVE:
            dx, dy = x - cx, y - cy
            
            if self.drag_center:
                self.settings.center_x = np.clip(x / self.cols, 0, 1)
                self.settings.center_y = np.clip(y / self.rows, 0, 1)

            elif self.drag_handle_up:
                # 1. Update Global Angle (Handle 1 defines Positive Normal)
                angle_rad = math.atan2(dy, dx)
                self.settings.angle_degree = np.rad2deg(angle_rad) + 90
                
                # 2. Update Upper Width (Distance)
                dist = math.sqrt(dx**2 + dy**2)
                old_total = self.settings.upper_sharp + self.settings.upper_trans
                if old_total > 0:
                    scale = dist / old_total
                    self.settings.upper_sharp = int(self.settings.upper_sharp * scale)
                    self.settings.upper_trans = int(self.settings.upper_trans * scale)
                    # Limit
                    self.settings.upper_sharp = max(10, self.settings.upper_sharp)
                    self.settings.upper_trans = max(10, self.settings.upper_trans)

            elif self.drag_handle_down:
                # 1. Update Global Angle (Handle 2 defines Negative Normal)
                # Logic: Handle 2 is opposite to Normal. So Normal angle is HandleAngle + 180
                angle_rad = math.atan2(dy, dx)
                # Angle of handle vector + 180 = Angle of Normal vector.
                # Line Angle = Normal Angle - 90.
                # So Line = (Handle + 180) - 90 = Handle + 90.
                self.settings.angle_degree = np.rad2deg(angle_rad) + 90
                
                # 2. Update Lower Width (Distance)
                dist = math.sqrt(dx**2 + dy**2)
                old_total = self.settings.lower_sharp + self.settings.lower_trans
                if old_total > 0:
                    scale = dist / old_total
                    self.settings.lower_sharp = int(self.settings.lower_sharp * scale)
                    self.settings.lower_trans = int(self.settings.lower_trans * scale)
                    # Limit
                    self.settings.lower_sharp = max(10, self.settings.lower_sharp)
                    self.settings.lower_trans = max(10, self.settings.lower_trans)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_center = False
            self.drag_handle_up = False
            self.drag_handle_down = False

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        # Trackbars (Simplified for visual adjustments)
        def nothing(x): pass
        cv2.createTrackbar("Blur", self.window_name, 50, 100, nothing)
        cv2.createTrackbar("Sat", self.window_name, 14, 30, nothing)
        cv2.createTrackbar("Cont", self.window_name, 11, 20, nothing)

        print("Controls:")
        print(" - Red Dot: Move Center.")
        print(" - Blue Dot 1: Rotate & Scale UPPER zone.")
        print(" - Blue Dot 2: Rotate & Scale LOWER zone.")

        while True:
            # Sync visual params
            self.blur_strength = cv2.getTrackbarPos("Blur", self.window_name)
            s = cv2.getTrackbarPos("Sat", self.window_name) / 10.0
            c = cv2.getTrackbarPos("Cont", self.window_name) / 10.0
            
            if abs(s - self.settings.saturation) > 0.01 or abs(c - self.settings.contrast) > 0.01:
                self.settings.saturation = s
                self.settings.contrast = c
                self.update_layers()
            
            # Check blur change (simple check)
            if hasattr(self, '_last_b') and self._last_b != self.blur_strength:
                self.update_layers()
            self._last_b = self.blur_strength

            # Render
            res = render_composite(self.cache_sharp, self.cache_blur, self.settings)
            final = draw_guides_cv2(res, self.settings)
            
            cv2.imshow(self.window_name, final)
            if cv2.waitKey(10) & 0xFF == 27: break # ESC
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LocalMiniatureApp("./test.jpg")
    app.run()
