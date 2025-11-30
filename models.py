from dataclasses import dataclass

@dataclass
class MiniatureSettings:
    """
    Data model for Asymmetric Tilt-Shift.
    Upper = Direction of the Normal Vector (Side A)
    Lower = Opposite Direction (Side B)
    """
    # Geometry
    center_x: float = 0.5
    center_y: float = 0.5
    angle_degree: float = 0.0
    
    # Side A (Upper / Positive Normal)
    upper_sharp: int = 100      # Distance from center to first dashed line
    upper_trans: int = 150      # Distance from dashed to solid line
    
    # Side B (Lower / Negative Normal)
    lower_sharp: int = 100      # Distance from center to first dashed line
    lower_trans: int = 150      # Distance from dashed to solid line
    
    # Visuals
    saturation: float = 1.0
    contrast: float = 1.0
