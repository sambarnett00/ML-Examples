from statistics import NormalDist
from typing import Tuple, List


class MapInfo:
    """Contains info about the start and checkpoints of a map"""
    start: Tuple[int, int]
    ckpts: List[Tuple[int, int, int, int]]

    def reset(self) -> None:
        self.start = None
        self.ckpts = []

    def scale(self, f: float) -> None:
        for i, (x1, y1, x2, y2) in enumerate(self.ckpts):
            self.ckpts[i] = (x1*f, y1*f, x2*f, y2*f)


# Mathematical functions, credit for the line_circle algorithm below
## https://www.jeffreythompson.org/collision-detection/line-circle.php

def line_circle(line: Tuple[float, float, float, float], circle: Tuple[float, float, float]) -> bool:
    """Returns true if line (x1, y1, x2, y2) is in collision with circle (cx, cy, r)"""
    x1, y1, x2, y2 = line
    cx, cy, r = circle

    if point_circle(x1, y1, cx, cy, r) or point_circle(x2, y2, cx, cy, r):
        return True

    d = dist(x1, y1, x2, y2)
    dot = ((cx-x1)*(x2-x1) + (cy-y1)*(y2-y1)) / d ** 2

    clx = x1 + dot * (x2-x1) ## closest x
    cly = y1 + dot * (y2-y1) ## closest y

    if not line_point(x1, y1, x2, y2, clx, cly):
        return False

    d = dist(clx, cly, cx, cy)
    return d <= r


def closest_dist(line, circle) -> float:
    """Returns the closest distance between the line (x1, y1, x2, y2) and circle (cx, cy, r)"""
    x1, y1, x2, y2 = line
    cx, cy, r = circle

    d = dist(x1, y1, x2, y2)
    dot = ((cx-x1)*(x2-x1) + (cy-y1)*(y2-y1)) / d ** 2

    clx = x1 + dot * (x2-x1) ## closest x
    cly = y1 + dot * (y2-y1) ## closest y

    return dist(clx, cly, cx, cy)


def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Distance between (x1, y1) (x2, y2)"""
    return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) ** 0.5


def point_circle(x: float, y: float, cx: float, cy: float, r: float) -> bool:
    """Returns True if (x, y) in circle (cx, cy, r)"""
    return dist(x, y, cx, cy) <= r


def line_point(x1: float, y1: float, x2: float, y2: float, px: float, py: float, buf: float = 0.1) -> bool:
    d1 = dist(px, py, x1, y1)
    d2 = dist(px, py, x2, y2)
    d = dist(x1, y1, x2, y2)
    
    return d1+d2 >= d-buf and d1+d2 <= d+buf


def get_z_value(p: float) -> float:
    """used to calculate the mut_z value from a p=MUT_RATE"""
    return NormalDist(mu=0, sigma=1).inv_cdf(1 - p)
