import cv2
# import pytesseract
import pandas as pd
import numpy as np
from ultralytics import YOLO
# from speed_estimation.tracker import *
from .tracker import *
from user_app.models import Record
import time
import math
from datetime import datetime
import easyocr
import msvcrt
print("EXISTING DATA")

 # Load the number plate detection model
best_path= r'C:\Users\Administrator\Desktop\Speed-detection-and-number-plate-recognition\speed_estimation\best.pt'
number_plate_model = YOLO(best_path)

#dictionary to store the best license late for each ID
best_license_plates ={}


# Preprocess the number plate region for OCR
def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu thresholding
    _, threshold_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshold_image

# # Perform OCR on the number plate region(py tesseract)
# def perform_ocr(image):
#     # Apply OCR using pytesseract
#     text = pytesseract.image_to_string(image, config='--psm 7 --oem 3')
#     return text

# Perform OCR on the number plate region(easyocr)
def perform_ocr(image):
    # Perform OCR on the thresholded image
    recognized_plates=[]
    reader = easyocr.Reader(['en', 'ne'])
    result = reader.readtext(image, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ', detail=0)
    recognized_plates = [''.join(result)]
    return recognized_plates


def license_plate(frame,y3,y4,x3,x4,id):
    # Perform number plate detection on the vehicle bounding box

        vehicle_image = frame[int(y3):int(y4), int(x3):int(x4)]
        # Perform object detection using YOLO
        detections = number_plate_model(vehicle_image)

        # Extract bounding boxes and crop number plate regions
        number_plate_boxes = []
        for detection in detections[0].boxes.data:
            if detection[5] == 0:
                number_plate_boxes.append(detection[:4])

        if len(number_plate_boxes) > 0:
            for box in number_plate_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(vehicle_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Crop and preprocess the number plate region
                cropped_image = vehicle_image[int(y1):int(y2), int(x1):int(x2)]
                #scaling image
                # scaled_image = cv2.resize(cropped_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

                # Preprocess the number plate region for OCR
                preprocessed_image = preprocess_image(cropped_image)
            
                # Perform OCR on the preprocessed number plate image
                number_plate_text = perform_ocr(preprocessed_image)
                prev_id = None
                best_plate = None
                # Check if the recognized plate is different from the previous plate
                if  id != prev_id:
                    best_plate = number_plate_text
                    prev_id = id
                    print("hello")

                print("Vehicle ID:", id, "Best Number Plate:", best_plate)
                #eta bata database ma update huna parne

        

#Calculating the axis position using the frame's shape and adjusting the percentage as needed
def calculate_axis_positions(frame):
    frame_width=frame.shape[1]
    frame_height = frame.shape[0]
    center_y1 = int(frame_height * 0.45)  
    center_y2 = int(frame_height * 0.75)  
    offset = int(frame_height * 0.02)  
    line_x1 = int(frame_width * 0)
    line_x2 = int(frame_width * 1)
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


def speed_calculation(frame, bbox_id, counter, vehicle_down, vehicle_up, center_y1, center_y2, offset, line_x1, line_x2, counter1,number_plate_model):
    number_plate_text=''#numberplate textholder
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2


        # # Perform number plate detection on the vehicle bounding box

        # vehicle_image = frame[int(y3):int(y4), int(x3):int(x4)]
        # # Perform object detection using YOLO
        # detections = number_plate_model(vehicle_image)

        # # Extract bounding boxes and crop number plate regions
        # number_plate_boxes = []
        # for detection in detections[0].boxes.data:
        #     if detection[5] == 0:
        #         number_plate_boxes.append(detection[:4])

        # if len(number_plate_boxes) > 0:
        #     for box in number_plate_boxes:
        #         x1, y1, x2, y2 = box
        #         cv2.rectangle(vehicle_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #         # Crop and preprocess the number plate region
        #         cropped_image = vehicle_image[int(y1):int(y2), int(x1):int(x2)]
        #         #scaling image
        #         # scaled_image = cv2.resize(cropped_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        #         # Preprocess the number plate region for OCR
        #         preprocessed_image = preprocess_image(cropped_image)
            
        #         # Perform OCR on the preprocessed number plate image
        #         number_plate_text = perform_ocr(preprocessed_image)
        #         prev_id = None
        #         best_plate = None
        #         # Check if the recognized plate is different from the previous plate
        #         if  id != prev_id:
        #             best_plate = number_plate_text
        #             prev_id = id
        #             print("hello")

        #         print("Vehicle ID:", id, "Best Number Plate:", best_plate)
        #         #eta bata database ma update huna parne

               


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
                    
                    #  # Check if the ID already has a best license plate
                    # if id not in best_license_plates:
                    #     # Call the license_plate() function to get the license plate for the ID
                    #     license_plate(frame,y3,y4,x3,x4,id,number_plate_model)
                   
                    frame = check_speed(a_speed_kh, bbox, frame)
                    license_plate(frame,y3,y4,x3,x4,id)
                    print("new data added")#update the records in database (database connection)
                    print("new data added++++++++++++++++")#update the records in database (database connection)
                    print(number_plate_text)
                    new_data = Record (
                    liscenseplate_no= number_plate_text,
                    speed= a_speed_kh,
                    date= datetime.now().date(),
                    IDs= 5,
                    count=len(counter)
                    )
                    new_data.save()


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
                    license_plate(frame,y3,y4,x3,x4,id)

                    print("new data added++++++++++++++++")#update the records in database (database connection)
                    print(number_plate_text)
                    new_data = Record(
                    liscenseplate_no= number_plate_text,
                    speed= a_speed_kh1,
                    date= datetime.now().date(),
                    IDs= 5,
                    count=len(counter1)
                    )
                    new_data.save()

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
    video_path = r'C:\Users\Administrator\Desktop\sufipuja.mp4'
    model_path = 'yolov8s.pt'
    class_list_path = r'C:\Users\Administrator\Desktop\Speed-detection-and-number-plate-recognition\speed_estimation\coco.txt'
   
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

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
            frame = cv2.resize(frame, (1020, 500))

            results = model.predict(frame)
            boxes = results[0].boxes.data
            df = pd.DataFrame(boxes).astype("float")
            object_list = []
            for index, row in df.iterrows():
                x1, y1, x2, y2, _, d = row
                c = class_list[int(d)]
                if c in ['car', 'motorcycle', 'truck', 'bus']:
                    object_list.append([x1, y1, x2, y2])

            center_y1, center_y2, offset,line_x1,line_x2 = calculate_axis_positions(frame)

            bbox_id = tracker.update(object_list)

            frame = speed_calculation(frame, bbox_id, counter, vehicle_down, vehicle_up, center_y1, center_y2, offset,line_x1,line_x2, counter1,number_plate_model)

            vehicle_down_count, vehicle_up_count = count_vehicles(counter, counter1)
            
            # print("NEW DATA: ")
            cv2.putText(frame, ('Count-') + str(vehicle_down_count + vehicle_up_count), (60, 90),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # cv2.imshow("Processed Video", frame)
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n'
        
        # key = cv2.waitKey(1)
        # if key == 27: #press Esc to exit
        #     break
        # elif key == ord('p') or key == ord('P'):#press P to pause
        #     paused = not paused
    
        if msvcrt.kbhit():
            key = msvcrt.getch()
            key = key.decode('utf-8')
        
            if key == 'q':
                break  # Exit the loop if 'q' is pressed
        
            if key == '\x1b':  # Escape key
                break  # Exit the loop if Escape is pressed
        
            if key == '\r':  # Enter key
                # Do something when Enter is pressed
                print("Enter key pressed.")
    
    
    cap.release()
    cv2.destroyAllWindows()