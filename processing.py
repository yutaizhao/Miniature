import cv2
import numpy as np
from typing import Tuple, List

from models import MiniatureSettings

class ImageToolbox:
    @staticmethod
    def apply_enhancements(image: np.ndarray, saturation: float, contrast: float) -> np.ndarray:
        """Applies Saturation and Contrast."""
        """This is an optional operation"""
        img = image.copy()
        if saturation != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
            (h, s, v) = cv2.split(hsv) #color/ saturation/ lightness
            s = np.clip(s * saturation, 0, 255)
            img = cv2.cvtColor(cv2.merge([h, s, v]).astype("uint8"), cv2.COLOR_HSV2BGR)
        
        if contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        return img

    @staticmethod
    def apply_blur_style(image: np.ndarray, level: int, shape: str) -> np.ndarray:
        """Generates blurred image."""
        if level <= 0: return image.copy()
        rescaled_level = max(1, int(level / 2)) #level : 0~100 for user
        k_size = min(rescaled_level * 2 + 1, 101)

        if shape == "gaussian":
            return cv2.GaussianBlur(image, (k_size, k_size), 0)
        elif shape == "box":
            return cv2.blur(image, (k_size, k_size))
        return cv2.GaussianBlur(image, (k_size, k_size), 0)


class TiltShiftZoneDefiner:
    """The class decides for each pixel whether it should be clear/blurred/intermediate"""
    def __init__(self, shape: Tuple[int, ...], settings: MiniatureSettings):
        self.rows, self.cols = shape[:2] # real image size
        self.settings = settings # center is (0.5,0.5) by default
        self.cx = int(settings.center_x * self.cols) # get the center (axis x) of the image
        self.cy = int(settings.center_y * self.rows) # get the center (axis y) of the image

    def _get_vectors(self, angle_deg):
        """Helper to get direction and normal vectors from an angle."""
        theta = np.deg2rad(angle_deg)
        # Normal vector (perpendicular to the zone line and it points to the blurred zone)
        # We assume the 'handle blue point' represents the Normal direction
        nx = np.cos(theta)
        ny = np.sin(theta)
        return nx, ny

    def compute_mask(self, downscale_factor=0.1) -> np.ndarray:
        """
        Calculates the mask with Downscaling Optimization.
        Result is upscaled back to original size.
        """
        
        """The following code generate X,Y"""
        """X,Y : Contain the coordinates of all points"""
        """So that in the futur : """
        """we dont need to loop on x and y to compute one by one the distance to each (x,y)"""
        """+ downscaling => Big acceleration !"""
        # 1. Determine calculation grid size (Low Res)
        calc_rows = int(self.rows * downscale_factor)
        calc_cols = int(self.cols * downscale_factor)
        
        # 2. Scale geometry parameters to low resolution
        scale_cx = self.cx * downscale_factor
        scale_cy = self.cy * downscale_factor
        
        X, Y = np.meshgrid(np.arange(calc_cols), np.arange(calc_rows))
        
        """Logic: Intersection of two half-planes"""
        # Part A: Upper Boundary
        # Calculate distance to the Upper Line
        # Project the vector (from center to pixel) onto the normal vector :
        # dotprod(dist_center_to_point, normal of upper line)
        # Note that the logic way is to :
        # Project the vector (from a point of upper line to pixel) onto the upper normal vector.
        # But later we are going to adjust this problem
        nx_up, ny_up = self._get_vectors(self.settings.angle_upper) # get the upper nomal
        dist_up = (X - scale_cx) * nx_up + (Y - scale_cy) * ny_up
        
        # Part B: Lower Boundary, doing the same
        nx_low, ny_low = self._get_vectors(self.settings.angle_lower)
        dist_low = (X - scale_cx) * nx_low + (Y - scale_cy) * ny_low
        
        """Compute the blurring/ decay factor of each zone """
        """0 : blurred, 1 : sharp, 0~1 : decay (what we want to compute) """
        # Initialize Mask as 1.0 (all clear)
        mask = np.ones((calc_rows, calc_cols), dtype=np.float32)
        def apply_decay(dist_field, sharp_lim, trans_lim):
            # sharp_lim : dist to sharp line
            # trans_lim : dist to intermdediate line
            # Scale limits to low resolution (since we used the downscaling)
            s_sharp = sharp_lim * downscale_factor
            s_trans = trans_lim * downscale_factor
            
            # Identify different zones : intermed zone and blur zone
            # No need for sharp zone : assuming sharp initially
            cond_trans = (dist_field > s_sharp) & (dist_field < (s_sharp + s_trans))
            cond_blur = (dist_field >= (s_sharp + s_trans))
            
            # cmputation of decay !!!!!!!! Linear interpolation !!!
            # 1 - (dist_to_pixel - lower bound)/ upper bound
            # we take the min in case upper and lower lines have intersection
            if s_trans > 0:
                mask[cond_trans] = np.minimum(mask[cond_trans], 1.0 - (dist_field[cond_trans] - s_sharp) / s_trans)
            
            mask[cond_blur] = 0.0

        # Apply constraints from both upper and lower lines
        apply_decay(dist_up, self.settings.dist_upper_sharp, self.settings.dist_upper_trans)
        apply_decay(dist_low, self.settings.dist_lower_sharp, self.settings.dist_lower_trans)
        
        # 3. Upscale mask to full resolution (Linear interpolation is fast and smooth enough)
        full_mask = cv2.resize(mask, (self.cols, self.rows), interpolation=cv2.INTER_LINEAR)
        return full_mask
        
    def compute_split_mask(self, downscale_factor=0.1) -> np.ndarray:
        """
        [NEW] Calculates a binary mask to distinguish Upper vs Lower regions.
        Result: 1.0 = Belongs to Upper Blur; 0.0 = Belongs to Lower Blur.
        """
        calc_rows = int(self.rows * downscale_factor)
        calc_cols = int(self.cols * downscale_factor)
        scale_cx = self.cx * downscale_factor
        scale_cy = self.cy * downscale_factor
        
        X, Y = np.meshgrid(np.arange(calc_cols), np.arange(calc_rows))
        
        # Calculate projection distance for both vectors
        nx_up, ny_up = self._get_vectors(self.settings.angle_upper)
        dist_up = (X - scale_cx) * nx_up + (Y - scale_cy) * ny_up
        
        nx_low, ny_low = self._get_vectors(self.settings.angle_lower)
        dist_low = (X - scale_cx) * nx_low + (Y - scale_cy) * ny_low
        
        # Logic: Whichever direction has a larger positive projection distance wins.
        # This effectively splits the image into two halves based on the handles.
        split_map = (dist_up > dist_low).astype(np.float32)
        
        # Resize to full resolution (Nearest Neighbor is fine for binary split)
        full_split = cv2.resize(split_map, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
        return full_split

    def get_handles_pos(self) -> List[Tuple[int, int]]:
        """Returns positions for drawing blue point handles."""
        # Handle 1 (Upper)
        nx_up, ny_up = self._get_vectors(self.settings.angle_upper)
        dist_up = self.settings.dist_upper_sharp
        h1 = (int(self.cx + nx_up * dist_up), int(self.cy + ny_up * dist_up))
        
        # Handle 2 (Lower)
        nx_low, ny_low = self._get_vectors(self.settings.angle_lower)
        dist_low = self.settings.dist_lower_sharp
        h2 = (int(self.cx + nx_low * dist_low), int(self.cy + ny_low * dist_low))
        
        return [h1, h2]

    def get_lines_geometry(self):
        """Returns lines data for drawing."""
        res = []
        for (angle, d_sharp, d_trans) in [
            (self.settings.angle_upper, self.settings.dist_upper_sharp, self.settings.dist_upper_trans),
            (self.settings.angle_lower, self.settings.dist_lower_sharp, self.settings.dist_lower_trans)
        ]:
            nx, ny = self._get_vectors(angle)
            # Perpendicular vector (for drawing the line across screen)
            dx, dy = -ny, nx
            res.append({
                'norm': (nx, ny),
                'dir': (dx, dy),
                'sharp': d_sharp,
                'total': d_sharp + d_trans
            })
        return res

""" After having the different zone : we do the rendering ! """
def render_composite(img_sharp: np.ndarray, img_blur: np.ndarray, settings: MiniatureSettings) -> np.ndarray:
    definer = TiltShiftZoneDefiner(img_sharp.shape, settings)
    mask = definer.compute_mask(downscale_factor=0.1)
    
    mask_3ch = cv2.merge([mask, mask, mask]) #same mask for RGB
    
    # We apply the mask on the image : Another linear interpolation !
    # For each pixel (x,y) : rgb = sharp_rgb * mask + blur_rgb * (1-mask)
    result = img_sharp * mask_3ch + img_blur * (1 - mask_3ch)
    return result.astype("uint8")
    
def render_composite_adv(img_sharp: np.ndarray,
                          img_blur_up: np.ndarray,
                          img_blur_low: np.ndarray,
                          settings: MiniatureSettings) -> np.ndarray:
    """
    [NEW] Advanced Rendering Pipeline.
    Blends: Sharp Image + Upper Blur Image + Lower Blur Image.
    """
    definer = TiltShiftZoneDefiner(img_sharp.shape, settings)
    
    # 1. Compute Blur Mask (0 to 1) : unchnaged like before
    mask_sharpness = definer.compute_mask(downscale_factor=0.1)
    
    # 2. Compute Split Mask (0 or 1)
    mask_split = definer.compute_split_mask(downscale_factor=0.1)
    
    # 3. Expand masks to 3 channels for RGB math
    mask_s_3ch = cv2.merge([mask_sharpness, mask_sharpness, mask_sharpness])
    mask_split_3ch = cv2.merge([mask_split, mask_split, mask_split])
    
    # 4. Composition
    combined_blur = img_blur_up * mask_split_3ch + img_blur_low * (1 - mask_split_3ch)
    result = img_sharp * mask_s_3ch + combined_blur * (1 - mask_s_3ch)
    
    return result.astype("uint8")

"""Draw guide lines : NOT VERY IMPORTANT"""
def draw_guides_cv2(image: np.ndarray, settings: MiniatureSettings, active_node: str = None) -> np.ndarray:
    overlay = image.copy()
    h, w = overlay.shape[:2]
    # Reduce drawing length calculation to avoid overflow issues, just needs to be "long enough"
    diag = 3000
    
    definer = TiltShiftZoneDefiner((h, w), settings)
    cx, cy = definer.cx, definer.cy
    lines = definer.get_lines_geometry()
    
    # Helper: Draw Dashed Line
    # Optimized: Draw segments instead of pixel-by-pixel loop for speed
    def draw_dashed_fast(p1, p2, color, thickness=1, dash=15):
        vec = np.array(p2) - np.array(p1)
        dist = np.linalg.norm(vec)
        if dist == 0: return
        unit_vec = vec / dist
        
        steps = int(dist / (dash * 2))
        for i in range(steps):
            start = np.array(p1) + unit_vec * (i * dash * 2)
            end = start + unit_vec * dash
            cv2.line(overlay, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)

    # Draw both boundaries (Upper and Lower are independent now)
    for line_info in lines:
        nx, ny = line_info['norm']
        dx, dy = line_info['dir']
        
        # Center of the sharp line
        ox_s, oy_s = cx + nx * line_info['sharp'], cy + ny * line_info['sharp']
        p1_s = (int(ox_s - dx * diag), int(oy_s - dy * diag))
        p2_s = (int(ox_s + dx * diag), int(oy_s + dy * diag))
        
        # Center of the blur line
        ox_b, oy_b = cx + nx * line_info['total'], cy + ny * line_info['total']
        p1_b = (int(ox_b - dx * diag), int(oy_b - dy * diag))
        p2_b = (int(ox_b + dx * diag), int(oy_b + dy * diag))
        
        draw_dashed_fast(p1_s, p2_s, (255, 255, 255), 2)
        cv2.line(overlay, p1_b, p2_b, (255, 255, 255), 2)

    # Helper: Point Drawing
    def draw_point(pos, color, is_active):
        radius = 8 if is_active else 6
        thick = 3 if is_active else 1
        cv2.circle(overlay, pos, radius, color, -1)
        cv2.circle(overlay, pos, radius+2, (255, 255, 255), thick)

    # 1. Center
    draw_point((cx, cy), (0, 0, 255), active_node == "center")
    
    # 2. Handles
    h_pos = definer.get_handles_pos()
    draw_point(h_pos[0], (255, 0, 0), active_node == "up")
    draw_point(h_pos[1], (255, 0, 0), active_node == "down")
    
    return overlay
