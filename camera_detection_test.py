# camera_detection_test.py
import pyrealsense2 as rs
import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

class YOLOCamera:
    def __init__(self, model_path="yolov8n.pt", target_class="person", confidence_threshold=0.4):
        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # YOLO model
        self.model = YOLO(model_path)
        self.target_class = target_class
        self.confidence_threshold = confidence_threshold

        # Class names (COCO dataset)
        self.classNames = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]

        self.distance_buffer = deque(maxlen=5)

    def run_person_detection_step(self):
        # Get frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None

        img = np.asanyarray(color_frame.get_data())
        timestamp = time.time()  # High-precision timestamp for sync

        person_detected = False
        smoothed_distance = None
        detection_info = {}  # For multi-person, but focus on closest for now

        results = self.model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = box.conf[0]
                if confidence > self.confidence_threshold:
                    cls = int(box.cls[0])
                    class_name = self.classNames[cls]

                    if class_name == self.target_class:
                        person_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Get depth from RealSense
                        distance = depth_frame.get_distance(cx, cy)
                        if distance == 0:  # sometimes depth is missing
                            continue

                        # Track per detection (could extend to multiple)
                        self.distance_buffer.append(distance)
                        smoothed_distance = np.mean(self.distance_buffer)
                        detection_info['distance'] = smoothed_distance
                        detection_info['bbox'] = (x1, y1, x2, y2)

                        # Draw box and label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(img, f"{class_name} {smoothed_distance:.2f}m", (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return {
            "person_detected": person_detected,
            "detection_info": detection_info,
            "distance": smoothed_distance if person_detected else None,
            "image": img,
            "timestamp": timestamp
        }

    def stop(self):
        self.pipeline.stop()