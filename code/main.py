import os
import csv
from collections import deque
import cv2
from ultralytics import YOLO
from utils.detector import BallDetector
from utils.draw_utils import draw_ball, draw_trajectory



def main(
    input_video,
    output_video,
    output_csv,
    model_name='yolo11l.pt',
    imgsz=640,
    stream=False,
    conf=0.4,
    # iou=0.5,
):
    """
    Main pipeline: detect and track ball in input_video and save annotated video and CSV.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_video}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Load YOLO model and BallTracker
    model = YOLO(model_name)             # pretrained on COCO
    # model.conf = conf                     # confidence threshold
    # # model.iou  = iou                    # NMS IoU threshold
    # model.classes = [32]                # restrict to sports ball (COCO class 32)
    
    # Initialize the BallDetector (detection-only)
    detector = BallDetector(model=model, imgsz=imgsz, stream=stream, conf=conf)


    # ROI boundaries (10% margin)
    # Define ROI from center of the frame
    roi_width = int(0.8 * width)   # 80% of frame width
    roi_height = int(0.8 * height) # 80% of frame height
    x_center, y_center = width // 2, height // 2

    x_min = x_center - roi_width // 2
    x_max = x_center + roi_width // 2
    y_min = y_center - roi_height // 2
    y_max = y_center + roi_height // 2
    
    frame_idx = 0

    # Keep last 30 trajectory points
    trajectory = deque(maxlen=30)
    
    # CSV rows
    rows = [['frame', 'x', 'y', 'visible']]

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        
        # Perform detection on the frame
        cx, cy, visible = detector.update(frame)
        
        # Only record and draw if ball is detected and inside ROI
        if visible and x_min <= cx <= x_max and y_min <= cy <= y_max:
            rows.append([frame_idx, int(cx), int(cy), 1])
            trajectory.append((int(cx), int(cy)))
            draw_ball(frame, (int(cx), int(cy))) # Draw bb on the frame

        else:
            # Ball not detected this frame
            rows.append([frame_idx, '', '', 0])
            # Do not append to trajectory

        # draw_trajectory(frame, trajectory)
        draw_trajectory(frame, list(trajectory))
        
        # Write annotated frame to output video
        writer.write(frame)
    
    # Release resources
    cap.release()
    writer.release()
    
    # Save annotations to CSV
    with open(output_csv, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerows(rows)

    print("Done.")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ball detection (frame-by-frame) with YOLO")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_video", type=str, default="output.mp4", help="Path to output annotated video")
    parser.add_argument("--output_csv", type=str, default="output.csv", help="Path to output CSV annotations")
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="YOLO model name or path")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size (px)")
    parser.add_argument("--stream", action="store_true", help="If set, YOLO returns generator")
    parser.add_argument("--conf", type=float, default=0.01, help="YOLO confidence threshold")
    args = parser.parse_args()

    main(
        input_video=args.input,
        output_video=args.output_video,
        output_csv=args.output_csv,
        model_name=args.model,
        imgsz=args.imgsz,
        stream=args.stream,
        conf=args.conf,
    )