from dataclasses import dataclass

@dataclass
class MiniatureSettings:
    """
    Data model for Fully Independent Tilt-Shift.
    Upper and Lower boundaries have their own Angles and Distances.
    """
    center_x: float = 0.5
    center_y: float = 0.5
    
    # Side A (Upper)
    angle_upper: float = 0.0    # Angle for the top line
    dist_upper_sharp: int = 50 # Distance to sharp line
    dist_upper_trans: int = 100 # Width of fade zone
    
    # Side B (Lower)
    angle_lower: float = 180.0  # Angle for the bottom line (default opposite)
    dist_lower_sharp: int = 50
    dist_lower_trans: int = 100
    
    # Visuals
    saturation: float = 1.0
    contrast: float = 1.0
