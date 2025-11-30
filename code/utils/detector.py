from ultralytics import YOLO
import numpy as np

class BallDetector:
    """
    Robust single-ball detection using YOLO.
    Chooses the most likely ball per frame based on:
    - YOLO confidence
    - Distance from previous frame (temporal consistency)

    returns center of first detected ball per frame.
    """
    def __init__(self, model: YOLO, imgsz: int = 640, stream: bool = False, conf: float = 0.4):
        """
        Args:
            model: YOLO model
            imgsz: Input image size for YOLO
            stream: If True, returns generator; else returns list
            conf: Confidence threshold for detections
        """
        self.model = model
        self.imgsz = imgsz
        self.stream = stream
        self.conf = conf
        self.prev_center = None  # last known ball position

    def update(self, frame):
        """
        Process a frame with YOLO detection only.
        Returns: (cx, cy, visible)
        """
        # Perform detection
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[32],   # Only detect sports ball
            stream=self.stream
        )

        # If no results, return not visible
        if len(results) == 0:
            return (0, 0, False)  # No detection (ball not visible)

        # results can be a list (stream=False) or generator (stream=True)
        # access the results of 1st img/frame
        result = results[0] # contains detected objects

        if not result.boxes: # If no boxes return not visible
            return (0, 0, False)

        # Extract class IDs and box coordinates for all detected objects
        boxes = result.boxes.xywh.cpu().numpy()    # [center_x, center_y, width, height]
        class_ids = result.boxes.cls.cpu().numpy().astype(int) # the ID of object Class;; To see the ID and Class label name RUN: result.names
        confidences = result.boxes.conf.cpu().numpy()

        # Filter only sports ball detections
        ball_indices = [i for i, cls in enumerate(class_ids) if cls == 32]
        if not ball_indices:
            return 0, 0, False

        # If previous position exists, choose detection closest to it
        if self.prev_center is not None:
            prev_cx, prev_cy = self.prev_center
            distances = [
                np.linalg.norm([boxes[i][0] - prev_cx, boxes[i][1] - prev_cy])
                for i in ball_indices
            ]
            # Combine confidence and distance
            alpha = 0.7  # weight for confidence
            beta = 0.3   # weight for distance (closer = better)
            scores = [
                alpha * confidences[i] + beta * (1 / (1 + distances[j]))
                for j, i in enumerate(ball_indices)
            ]
            best_idx = ball_indices[np.argmax(scores)]
        else:
            # No previous position: pick highest confidence
            best_idx = ball_indices[np.argmax([confidences[i] for i in ball_indices])]

        cx, cy = float(boxes[best_idx][0]), float(boxes[best_idx][1])
        self.prev_center = (cx, cy)

        return cx, cy, True
    

        # # Find first sports ball (class 32)
        # for i, cls in enumerate(class_ids):
        #     if cls == 32:
        #         cx, cy = boxes[i][0], boxes[i][1]
        #         return (float(cx), float(cy), True)

        # # If no ball detected
        # return (0, 0, False)