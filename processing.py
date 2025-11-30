import cv2
import numpy as np
from typing import Tuple, List

from models import MiniatureSettings

class ImageToolbox:
    @staticmethod
    def apply_enhancements(image: np.ndarray, saturation: float, contrast: float) -> np.ndarray:
        """Applies Saturation and Contrast."""
        img = image.copy()
        if saturation != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
            (h, s, v) = cv2.split(hsv)
            s = np.clip(s * saturation, 0, 255)
            img = cv2.cvtColor(cv2.merge([h, s, v]).astype("uint8"), cv2.COLOR_HSV2BGR)
        
        if contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        return img

    @staticmethod
    def apply_blur_style(image: np.ndarray, level: int, shape: str) -> np.ndarray:
        """Generates blurred image."""
        if level <= 0: return image.copy()
        mapped = max(1, int(level / 2))
        k_size = min(mapped * 2 + 1, 101)

        if shape == "gaussian":
            return cv2.GaussianBlur(image, (k_size, k_size), 0)
        elif shape == "box":
            return cv2.blur(image, (k_size, k_size))
        return cv2.GaussianBlur(image, (k_size, k_size), 0)


class TiltShiftZoneDefiner:
    def __init__(self, shape: Tuple[int, ...], settings: MiniatureSettings):
        self.rows, self.cols = shape[:2]
        self.settings = settings
        
        # Geometry Vectors
        self.theta = np.deg2rad(settings.angle_degree)
        self.cx = int(settings.center_x * self.cols)
        self.cy = int(settings.center_y * self.rows)
        
        # Direction (Along the line)
        self.dir_vec = np.array([np.cos(self.theta), np.sin(self.theta)])
        # Normal (Perpendicular, pointing to Side A)
        self.norm_vec = np.array([-np.sin(self.theta), np.cos(self.theta)])

    def compute_mask(self) -> np.ndarray:
        """
        Calculates the Asymmetric Alpha Mask.
        Distinguishes between Positive side (Upper) and Negative side (Lower).
        """
        X, Y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        
        # Signed Distance Calculation
        # Positive values = Side A, Negative values = Side B
        signed_dist = (X - self.cx) * self.norm_vec[0] + (Y - self.cy) * self.norm_vec[1]
        
        mask = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # --- Logic for Side A (Upper, dist > 0) ---
        sw_up = self.settings.upper_sharp
        tw_up = self.settings.upper_trans
        
        # Pixels on Side A
        mask_A = (signed_dist >= 0)
        dist_A = signed_dist[mask_A]
        
        # 1. Sharp Area A
        sharp_A = dist_A < sw_up
        # 2. Transition Area A
        trans_A = (dist_A >= sw_up) & (dist_A < (sw_up + tw_up))
        
        # Fill Mask A
        # We need to map these boolean indices back to the main mask
        # This is a bit tricky with numpy boolean indexing, so let's do it per zone.
        
        # --- Simplified Logic using np.where for readability ---
        
        # 1. Sharp Zone (Both sides)
        # Upper sharp
        mask[(signed_dist >= 0) & (signed_dist < sw_up)] = 1.0
        # Lower sharp (using abs for negative side)
        sw_low = self.settings.lower_sharp
        mask[(signed_dist < 0) & (signed_dist > -sw_low)] = 1.0
        
        # 2. Transition Zone Upper
        if tw_up > 0:
            cond_up = (signed_dist >= sw_up) & (signed_dist < (sw_up + tw_up))
            mask[cond_up] = 1.0 - (signed_dist[cond_up] - sw_up) / tw_up
            
        # 3. Transition Zone Lower
        tw_low = self.settings.lower_trans
        if tw_low > 0:
            # Distance is negative, e.g., -150. abs is 150.
            abs_dist = -signed_dist
            cond_low = (signed_dist <= -sw_low) & (signed_dist > -(sw_low + tw_low))
            mask[cond_low] = 1.0 - (abs_dist[cond_low] - sw_low) / tw_low
            
        return mask

    def get_handles_pos(self) -> List[Tuple[int, int]]:
        """
        Returns two points: [Handle_Upper, Handle_Lower]
        """
        # Handle 1: Upper limit (Positive Normal)
        total_up = self.settings.upper_sharp + self.settings.upper_trans
        h1x = self.cx + self.norm_vec[0] * total_up
        h1y = self.cy + self.norm_vec[1] * total_up
        
        # Handle 2: Lower limit (Negative Normal)
        total_low = self.settings.lower_sharp + self.settings.lower_trans
        h2x = self.cx - self.norm_vec[0] * total_low
        h2y = self.cy - self.norm_vec[1] * total_low
        
        return [(int(h1x), int(h1y)), (int(h2x), int(h2y))]


def render_composite(img_sharp: np.ndarray, img_blur: np.ndarray, settings: MiniatureSettings) -> np.ndarray:
    definer = TiltShiftZoneDefiner(img_sharp.shape, settings)
    mask = definer.compute_mask()
    mask_3ch = cv2.merge([mask, mask, mask])
    result = (img_sharp * mask_3ch + img_blur * (1 - mask_3ch)).astype("uint8")
    return result


def draw_guides_cv2(image: np.ndarray, settings: MiniatureSettings) -> np.ndarray:
    """Draws asymmetric guides and two handles."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    diag = int((h**2 + w**2)**0.5)
    
    definer = TiltShiftZoneDefiner((h, w), settings)
    cx, cy = definer.cx, definer.cy
    nx, ny = definer.norm_vec
    dx, dy = definer.dir_vec
    
    def get_line_pts(offset):
        ox, oy = cx + nx * offset, cy + ny * offset
        p1 = (int(ox - dx * diag), int(oy - dy * diag))
        p2 = (int(ox + dx * diag), int(oy + dy * diag))
        return p1, p2

    def draw_dashed(p1, p2):
        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        if dist == 0: return
        vx, vy = (p2[0]-p1[0])/dist, (p2[1]-p1[1])/dist
        curr = 0
        while curr < dist:
            a = (int(p1[0]+vx*curr), int(p1[1]+vy*curr))
            b = (int(p1[0]+vx*min(curr+10, dist)), int(p1[1]+vy*min(curr+10, dist)))
            cv2.line(overlay, a, b, (255, 255, 255), 2)
            curr += 20

    # --- Upper Side (Positive) ---
    # Dashed (Sharp Limit)
    p1, p2 = get_line_pts(settings.upper_sharp)
    draw_dashed(p1, p2)
    # Solid (Blur Limit)
    p1, p2 = get_line_pts(settings.upper_sharp + settings.upper_trans)
    cv2.line(overlay, p1, p2, (255, 255, 255), 2)

    # --- Lower Side (Negative) ---
    # Dashed (Sharp Limit)
    p1, p2 = get_line_pts(-settings.lower_sharp)
    draw_dashed(p1, p2)
    # Solid (Blur Limit)
    p1, p2 = get_line_pts(-(settings.lower_sharp + settings.lower_trans))
    cv2.line(overlay, p1, p2, (255, 255, 255), 2)

    # --- Interactive Points ---
    # Center (Red)
    cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
    
    # Handles (Blue)
    h_pos = definer.get_handles_pos()
    
    # Handle 1 (Upper)
    cv2.circle(overlay, h_pos[0], 6, (255, 0, 0), -1) # Blue filled
    cv2.circle(overlay, h_pos[0], 8, (255, 255, 255), 1) # White border
    
    # Handle 2 (Lower)
    cv2.circle(overlay, h_pos[1], 6, (255, 0, 0), -1) # Blue filled
    cv2.circle(overlay, h_pos[1], 8, (255, 255, 255), 1) # White border

    return overlay
