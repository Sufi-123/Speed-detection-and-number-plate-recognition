import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from user_app.models import vehicle
#yolo model loaded
model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)               

#VideoLink
cap=cv2.VideoCapture(r'C:\Users\sufid\Desktop\speeddetection&numberplate\Speed-detection-and-number-plate-recognition\Cars_Moving.mp4')

my_file = open(r"C:\Users\sufid\Desktop\Speed-detection-and-number-plate-recognition\speed_estimation\coco.txt", "r")
data = my_file.read()
class_detections = data.split("\n")

count=0
paused = False

#Region of Intrest
area = [(301,227),(529,227),(526,398),[12,398]]
area_limits =[301,227,529,227]

#Tracking Vehicles:
tracker = Tracker()
vehicles_entering ={}
counted_ids = set()
id_counter =1 


while True:
    if not paused:
        ret,frame = cap.read()
        if not ret:
            break

        #frame resized
        frame=cv2.resize(frame,(1020,500))                      

        results=model.predict(frame)
        a= results[0].boxes.boxes
       
        px=pd.DataFrame(a).astype("float")
        detections=[]
        detected_ids =set()
        for index, row in px.iterrows():

            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=class_detections[d]

            if c in ['car','motorcycle','truck','bus']:
                detections.append([x1,y1,x2,y2])
                detected_ids.add(index)

        bbox_id=tracker.update(detections)

        cv2.line(frame, (301,227),(529,227),(0,0,255),5)

        for bbox in bbox_id:
            x3,y3,x4,y4,id=bbox 
            w,h = x4 - x3 , y4 - y3

            #center point of the vehicle
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2

          
        
            if area_limits[0] < cx < area_limits[2] and area_limits[1] - 15 <cy < area_limits[1] +15:
                if id not in counted_ids:
                    if id not in vehicles_entering:
                        vehicles_entering[id] = id_counter                            
                        id_counter +=1

                    counted_ids.add(id)

                    cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                    cv2.putText(frame,str(vehicles_entering[id]),(x3,y3-10),0,0.75,(255,255,255),2)

                    
                    a=vehicle.objects.get(ID= counted_ids)
                    a.count=vehicles_entering
                    a.save()


    
        # print(type(frame))
        cv2.imshow("RGB", frame)

    key= cv2.waitKey(1)
    if key==27: #press Esc to exit
        break
    elif key == ord('p') or key== ord('P'): #Press P  to pause/resume the video
        paused = not paused



cap.release()
cv2.destroyAllWindows()




