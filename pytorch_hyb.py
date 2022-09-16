import random
import numpy as np
from sys import path_hooks
import time
import cv2
import torch
from PIL import Image
from faster_video import FileVideoStream
from threading import Thread
import threading
import matplotlib.pyplot as plt

from pathlib import Path
# Model

def detect(model, video_path, show = True):
    cap = cv2.VideoCapture(video_path)
    i = 0
    boxes = []
    while(True):
        i += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        if i % 10 == 0:
            result = model(frame[:,:,::-1])
            boxes = result.pandas().xyxy[0]
            boxes = boxes.iloc[:,:4].values
            # frame.render()
            # frame = frame.imgs[0]
            # frame = frame[:,:,::-1]
        if show:
            for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
                xA, yA, xB, yB = list(map(int, [xA, yA, xB, yB]))
                cv2.rectangle(frame, (xA, yA), (xB, yB),
                                (0, 255, 0), 2)
            # Display the resulting frame
            frame = cv2.resize(frame, (640,480))
            frame = cv2.putText(frame,f'Number of people {len(boxes)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,225))
            cv2.imshow(str(video_path),frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # print(f'{video_path} {len(boxes)}')
            pass
        

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]
model.cpu()

video_paths = list(Path('videos').glob('*'))

try:
    start = time.time()
    thread_list = [threading.Thread(target=detect,args=(model,str(video_path),True)) for video_path in video_paths]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    end = time.time()
    print('Time taken: ', end-start)
except:
    print('Error: unable to start thread')