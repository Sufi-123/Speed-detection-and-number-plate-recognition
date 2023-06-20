import cv2
import easyocr
from IPython.display import Image, display

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('best.pt') 
model.conf = 0.4  # Confidence threshold for detection

# Initializing OCR
reader = easyocr.Reader(['en', 'ne'])

def recognize_number_plate(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Initialize an empty list to store recognized number plates
    recognized_plates = []
    
    # Read frames from the video
    while True:
        ret, frame = video.read()
        
        # Check if a frame was successfully read
        if not ret:
            break
        
        # Perform object detection using YOLO
        detections = model(frame)
        
        # Extract bounding boxes and crop number plate regions
        number_plate_box = None
        for detection in detections[0].boxes.data:
            if detection[5] == 0:  
                number_plate_box = detection[:4]
                break
        
        # Crop the number plate region
        if number_plate_box is not None:
            x1, y1, x2, y2 = number_plate_box
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # # Display the cropped image
            # display(Image(data=cv2.imencode('.jpg', cropped_image)[1].tobytes()))
            
            # Perform OCR on the number plate image
            result = reader.readtext(cropped_image, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ', detail=0)
            
            # Extract the text from the OCR result
            recognized_plate = ''.join(result)
            
            # Add the recognized plate to the list
            recognized_plates.append(recognized_plate)
    
    # Release the video file
    video.release()
    
    return recognized_plates

# Path to video file
video_path = 'example.mp4'  
recognized_plates = recognize_number_plate(video_path)

# Print the recognized number plates
for plate in recognized_plates:
    print(plate)
