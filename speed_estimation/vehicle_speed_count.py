import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from .tracker import *

import time
import math

#Calculating the axis position using the frame's shape and adjusting the percentage as needed
def calculate_axis_positions(frame):
    frame_width=frame.shape[1]
    frame_height = frame.shape[0]
    center_y1 = int(frame_height * 0.55)  
    center_y2 = int(frame_height * 0.8)  
    offset = int(frame_height * 0.02)  
    line_x1 = int(frame_width * 0.1)
    line_x2 = int(frame_width * 0.9)
    return center_y1, center_y2, offset,line_x1,line_x2

#Checking whether the speed is above the speed limit or not
def check_speed(speed, bbox, frame):
    x3, y3, x4, y4, id = bbox
    cx = int(x3 + x4) // 2
    cy = int(y3 + y4) // 2

    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.putText(frame, str(int(speed)) + 'Km/h', (int(x4), int(y4)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)  # Set rectangle color to green

    if speed > 50:
        cv2.putText(frame, str(int(speed)) + 'Km/h', (int(x4), int(y4)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                    (0, 0, 255), 2)  # Set text color to red
        cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)  # Set rectangle color to red

    return frame

#Calculating the speed of the vehicle
def speed_calculation(frame, bbox_id, counter, vehicle_down, vehicle_up, center_y1, center_y2, offset,line_x1,line_x2,counter1):
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        #cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 255), 6, 1)

        if center_y1 < (cy + offset) and center_y1 > (cy - offset):
            vehicle_down[id] = time.time()
        if id in vehicle_down:
            if center_y2 < (cy + offset) and center_y2 > (cy - offset):
                elapsed_time = time.time() - vehicle_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    # Calculate distance dynamically based on coordinates
                    distance = abs(y4 - y3)  # Use the height of the bounding box as the distance
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.putText(frame, str(len(counter)), (int(x3), int(y3)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
                    frame = check_speed(a_speed_kh, bbox, frame)

        if center_y2 < (cy + offset) and center_y2 > (cy - offset):
            vehicle_up[id] = time.time()
        if id in vehicle_up:
            if center_y1 < (cy + offset) and center_y1 > (cy - offset):
                elapsed1_time = time.time() - vehicle_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    # Calculate distance dynamically based on coordinates
                    distance1 = abs(y4 - y3)  # Use the height of the bounding box as the distance
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.putText(frame, str(len(counter1)), (int(x3), int(y3)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
                    frame = check_speed(a_speed_kh1, bbox, frame)

    cv2.line(frame, (line_x1, center_y1), (line_x2, center_y1), (255, 255, 255), 1)
    cv2.line(frame, (line_x1, center_y2), (line_x2, center_y2), (255, 255, 255), 1)

    vehicle_down_count = len(counter)
    vehicle_up_count = len(counter1)

    cv2.putText(frame, ('Count-') + str(vehicle_down_count + vehicle_up_count), (60, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    return frame


def count_vehicles(counter, counter1):
    vehicle_down_count = len(counter)
    vehicle_up_count = len(counter1)
    return vehicle_down_count, vehicle_up_count


def process_video():
    video_path = r'C:\Users\Puja\Desktop\aiproject\neha\Detection\obj_detection\video.mp4'
    model_path = 'yolov8s.pt'
    class_list_path = r'C:\Users\Puja\Desktop\aiproject\neha\Detection\obj_detection\coco.txt'

    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    with open(class_list_path, "r") as f:
        class_list = f.read().split("\n")

    tracker = Tracker()
    counter = []
    vehicle_down = {}
    counter1 = []
    vehicle_up = {}
    paused = False
    count = 0
    while True:
        if not paused:
            ret, frame = cap.read(0)
            if not ret:
                break
            #Frame skipping(every 3rd frame is being processed)
            # count += 1
            # if count % 3 != 0:
            #     continue

            frame = cv2.resize(frame, (1020, 500))

            results = model.predict(frame)
            boxes = results[0].boxes.boxes
            df = pd.DataFrame(boxes).astype("float")
            object_list = []
            for index, row in df.iterrows():
                x1, y1, x2, y2, _, d = row
                c = class_list[int(d)]
                if c in ['car', 'motorcycle', 'truck', 'bus']:
                    object_list.append([x1, y1, x2, y2])

            center_y1, center_y2, offset,line_x1,line_x2 = calculate_axis_positions(frame)

            bbox_id = tracker.update(object_list)

            frame = speed_calculation(frame, bbox_id, counter, vehicle_down, vehicle_up, center_y1, center_y2, offset,line_x1,line_x2, counter1)

            vehicle_down_count, vehicle_up_count = count_vehicles(counter, counter1)

            cv2.putText(frame, ('Count-') + str(vehicle_down_count + vehicle_up_count), (60, 90),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # cv2.imshow("Processed Video", frame)
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n'

        key= cv2.waitKey(1)
        if key==27: #press Esc to exit
            break
        elif key == ord('p') or key == ord('P'):#press P to pause
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


# Run the video processing
# video_path = r'C:\Users\Puja\Desktop\aiproject\neha\Detection\obj_detection\video.mp4'
# model_path = 'yolov8s.pt'
# class_list_path = r'C:\Users\Puja\Desktop\aiproject\neha\Detection\obj_detection\coco.txt'

