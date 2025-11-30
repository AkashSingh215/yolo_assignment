import cv2
import numpy as np

def draw_ball(frame, center, color=(0,0,255), radius=5, thickness=-1): # red color
    """
    Draws a circle at the given center on the frame.
    """
    cv2.circle(frame, center, radius, color, thickness)


def draw_trajectory(frame, points, color=(0, 255, 0), thickness=2):
    if len(points) < 2:
        return

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, thickness)