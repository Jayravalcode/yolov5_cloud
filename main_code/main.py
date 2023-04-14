import os
import cv2
import numpy as np
import torch
import requests
import time

# Define the directories to save images and blur detection results
image_dir = "Images"
output_folder = "Output"

# Create the directories if they don't exist
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the YOLOv7 model and its hyperparameters
# model = torch.hub.load('.', 'custom', 'yolov7.pt', source='local')  
model = torch.hub.load('Jayravalcode/yolov5_cloud', 'yolov5s', pretrained=True)
conf_thres = 0.75
iou_thres = 0.5


# Create the directories if they don't exist

with open("urls.txt", "r") as f:
    URLs = [line.strip() for line in f]
# Continuously wait for new images in the input folder
while True:

    for URL in URLs:
        r = requests.get(url=URL)
        data = r.json()
        did = list(data["data"]["live-record"].keys())

        rtmp = []
        for i in did:
            rtmp.append("rtmp://" + URL.split("/")[2].split(":")[0] + ":80/live-record/" + i)

        for rtmp_link in rtmp:
            # Define the time gap between frame captures (in seconds)
            time_gap = 1

            # Capture frames from the RTMP link and save them in the image directory
            cap = cv2.VideoCapture(rtmp_link)
            for i in range(1):
                ret, frame = cap.read()
                if ret:
                    # Get the current timestamp and format it
                    timestamp = time.time()
                    timestamp_str = time.strftime(f"{rtmp_link.split('/')[4]}_%H:%M:%S", time.localtime(timestamp))

                    # Save the image in the image directory
                    image_filename = f"{timestamp_str}.jpg"
                    image_path = os.path.join(image_dir, image_filename)
                    cv2.imwrite(image_path, frame)

                    image_files = [f"./Images/"+f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.jpg')]
                    results = model(image_files)
                    # results.save("Output")
                    detections = results.pred[0][results.pred[0][:, 4] > conf_thres]
                    print("==================")
                    print("Results:-",results.pred[0][:,4])                    
                    print(detections)
                    print("length of detections:-",len(detections),",","Checker:=>RTMP Link" + " " + rtmp_link + " " + "is done")
                    print("==================")
                    if len(detections) > 0:
                        results.save("Output")
                    os.system(f"rm -r ./Images/*")
                    i = i+1
                else:
                    break   
                            #time.sleep(180)
            cap.release()
        print("RTMP Link" + " " + rtmp_link + " " + "is done")
    
        

