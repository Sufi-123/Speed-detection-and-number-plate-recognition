import re
import uuid
import cv2
import pandas as pd
import numpy as np
from django.contrib.auth.decorators import login_required
from .tracker import *
from user_app.models import Record, Station
import time
import math
from datetime import datetime
import requests
from matplotlib import pyplot as plt
import ultralytics
from ultralytics import YOLO
import functools
from keras.models import load_model
# from keras.preprocessing import image
# import msvcrt

#retrieving the mac_address
mac_address = (':'.join(re.findall('..', '%012x' % uuid.getnode())))


print("EXISTING DATA")

 # Load the number plate detection model
best_path= r'models\best.pt'
number_plate_model = YOLO(best_path)

# Load the character recognition model
model= load_model(r'models\model.h5')

# Preprocess the number plate region for OCR
def preprocess_image(image,scale_factor=3):
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
     # Convert to grayscale
    grayscale_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    #gaussian blur
    blur= cv2.GaussianBlur(grayscale_image,(5,5),0)
    
    return blur

def threshold(image):
    # Apply Otsu thresholding
    _, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Invert the colors
    inverted_image = cv2.bitwise_not(threshold_image)
    return inverted_image  

def segment_characters(image):
    # Perform connected components analysis on the thresholded image and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv2.connectedComponents(image)
    mask = np.zeros(image.shape, dtype="uint8")
    
    # Set lower bound and upper bound criteria for characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 100 # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 10 # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 255:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(image.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound, 
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)
    return mask


def predict_character(image, model):
    # Resize and preprocess the image
    image = cv2.resize(image, (50, 50))
    image = image.reshape(-1, image.shape[0], 50, 1)

    # Predict using the CNN model
    classes = model.predict(image)
    classes = np.argmax(classes)  # Get the index of the maximum prediction

    return classes

def license_plate(frame,y3,y4,x3,x4,id):
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
            preprocessed_image = preprocess_image(cropped_image)
            thresholded_image = threshold(preprocessed_image)
            mask = segment_characters(thresholded_image)
            # Find contours and get bounding box for each contour
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]

            # Find contours and get bounding box for each contour
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                
                # Sort the bounding boxes from left to right, top to bottom
                # sort by Y first, and then sort by X if Ys are similar
            def compare(rect1, rect2):
                if abs(rect1[1] - rect2[1]) > 10:
                    return rect1[1] - rect2[1]
                else:
                    return rect1[0] - rect2[0]
            boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )
            # Draw bounding boxes on the mask image
            mask_with_boxes = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)  # Convert mask to BGR color space
            segmented_characters = []
            for (x, y, w, h) in boundingBoxes:
                # Extract the segmented character region from the mask
                segmented_character = mask[y:y+h, x:x+w]
                # Add the segmented character to the list
                segmented_characters.append(segmented_character)
                # Draw the bounding box on the mask image
                cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Iterate over segmented characters and recognize them
            plate = ''
            for character in segmented_characters:
                cnn_prediction = predict_character(character, model)

                if cnn_prediction < 10:
                    plate += str(cnn_prediction)
                elif cnn_prediction == 10:
                    plate += 'BA '
                elif cnn_prediction == 11:
                    plate += 'CHA '
                elif cnn_prediction == 12:
                    plate += 'PA '
                else:
                    plate += 'Unknown'
            
            prev_id = None
            best_plate = None
#           # Check if the plate is different from the previous plate
            if  id != prev_id:
                best_plate = plate
                prev_id = id


            # Print the recognized number plate
            print('vehicle ID:', id,'Plate:', best_plate)   
            return best_plate

#Calculating the axis position using the frame's shape and adjusting the percentage as needed
def calculate_axis_positions(frame):
    frame_width=frame.shape[1]
    frame_height = frame.shape[0]
    center_y1 = int(frame_height * 0.35) 
    # 0.2..0.4   47  35  62
    center_y2 = int(frame_height * 0.62)  
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


def speed_calculation(frame, bbox_id, counter, vehicle_down, vehicle_up, center_y1, center_y2, offset, line_x1, line_x2, counter1,fps ):
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        distance_known = 10 #Assumed distance between the camera and vehicle
        height_known = 2 #Assumed height of the vehicle

        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        if center_y1 < (cy + offset) and center_y1 > (cy - offset):
            vehicle_down[id] = time.time()
           
        if id in vehicle_down:
            
            if center_y2 < (cy + offset) and center_y2 > (cy - offset):
                elapsed_time =1/fps
                if counter.count(id) == 0:
                    counter.append(id)
                    # Calculate distance dynamically based on coordinates
                    distance = distance_known * height_known / (y4 - y3)  # Use the height of the bounding box as the distance
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.putText(frame, str(len(counter)), (int(x3), int(y3)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2) 
                    frame = check_speed(a_speed_kh, bbox, frame)
                    best_plate= license_plate(frame,y3,y4,x3,x4,id)
                    
                    print("new data added")#update the records in database (database connection)
                    print("new data added++++++++++++++++")#update the records in database (database connection)
                    
                    new_data = Record(
                   
                    stationID = Station.objects.get(mac_address=mac_address),  # Rerieve Station user using mac_address
                    speed= a_speed_kh,
                    date= datetime.now().date(),
                    count=len(counter),
                    liscenseplate_no= best_plate,
                    )
                    new_data.save()
         
        if center_y2 < (cy + offset) and center_y2 > (cy - offset):
            vehicle_up[id] = time.time()
        if id in vehicle_up:
            if center_y1 < (cy + offset) and center_y1 > (cy - offset):
                elapsed1_time = 1 / fps
                if counter1.count(id) == 0:
                    counter1.append(id)
                    # Calculate distance dynamically based on coordinates
                    distance1 = distance_known * height_known / (y4 - y3)  # Use the height of the bounding box as the distance
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.putText(frame, str(len(counter1)), (int(x3), int(y3)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
                   
                    frame = check_speed(a_speed_kh1, bbox, frame)
                    best_plate=license_plate(frame,y3,y4,x3,x4,id)

                    print("new data added++++++++++++++++")#update the records in database (database connection)
                    # print(number_plate_text)
                    
                    new_data = Record(
                    stationID = Station.objects.get(mac_address=mac_address),  
                    speed= a_speed_kh1,
                    date= datetime.now().date(),
                    count=len(counter1),
                    liscenseplate_no= best_plate,
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
    video_path = r'speed_estimation\test_vid\IMG-2732.mp4'
    model_path = 'models\yolov8s.pt'
    class_list_path = r'models\coco.txt'
   
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
    fps = 25
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

            frame = speed_calculation(frame, bbox_id, counter, vehicle_down, vehicle_up, center_y1, center_y2, offset,line_x1,line_x2, counter1,fps)

            vehicle_down_count, vehicle_up_count = count_vehicles(counter, counter1)
            
            # print("NEW DATA: ")
            cv2.putText(frame, ('Count-') + str(vehicle_down_count + vehicle_up_count), (60, 90),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # cv2.imshow("Processed Video", frame)
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n'

    
    cap.release()
    cv2.destroyAllWindows()