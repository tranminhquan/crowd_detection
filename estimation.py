import os
import streamlit as st
import cv2
from stream import show_sidebar, show_drawing, show_results, show_settings
import numpy as np
from detect import count_base_motion, detect, get_union_area, reid_base_boxes, count_people_in_queue, speed_track
from util import *
from optical_flow import OpticalFlow
import torch

from statsmodels.tsa.stattools import adfuller

import time

ROOT_PATH = os.path.join(os.getcwd(), 'videos')

def get_images(video_path):
    #read first frame of video with cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame

def main():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/data/jupyter/maindata/mle_duytran/crow-human/crowdhuman_yolov5m.pt')  
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.classes = [0]
    # state = SessionState.get(
    #     upload_key=None, enabled=True, start=False, run=False)
    state = st.session_state
    st.write("### Choose videos")
    path = st.multiselect("", os.listdir(ROOT_PATH))
    path = [os.path.join(ROOT_PATH, p) for p in path]

    drawing_mode, stroke_width, stroke_color, skip_frame, runtime_type = show_sidebar.show()
    if runtime_type == 'GPU':
        model.cuda()
    else:
        model.cpu()

    images = []
    areas_info = {}
    areas_draw = {}
    areas_crowd = {}
    lines_info = {}
    motion_detectors = {}
    queues_info = {}
    queue_in_area = {}
    queue_optical_flow = {}
    queue_distance = {}
    for video_path in path:
        image = get_images(video_path)
        # Show the image with streamlit canvas
        image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB)
        print('shpae', image.shape)
        scale_width = image.shape[1] / 640
        scale_height = image.shape[0] / 480
        images.append(image)

        areas_info[video_path], areas_draw[video_path],areas_crowd[video_path],lines_info[video_path],queues_info[video_path] = show_drawing.show(stroke_width, stroke_color, image, drawing_mode, scale_width, scale_height, None, key = video_path)
        # check if polygon queue inside rectangle area
        for area_name, area in areas_info[video_path].items():
            area = list(map(int, area))
            queue_in_area[area_name] = []
            for queue_name, queue in queues_info[video_path].items():
                for point in queue:
                    if area[0] < point[0] < area[2] and area[1] < point[1] < area[3]:
                        queue_in_area[area_name].append(queue_name)
                        break
        for area_name, area in areas_info.items():
            motion_detectors[area_name] = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    print('queue_in_area', queue_in_area)
    print('skip_frame', skip_frame)
    st.spinner('Testing...')
    if st.button("Start setting", key = 'start_proccess'):
     
        state.start = True
        state.run = False
        state.show_frames = False
        caps = {}
        fps = {}
        for video_path in path:
            if video_path not in caps:
                caps[video_path] = cv2.VideoCapture(video_path)
                fps[video_path] = caps[video_path].get(cv2.CAP_PROP_FPS)
        state.caps = caps
        state.process = True
        placeholder = st.empty()
        results = {'time': [],'area': [], 'count': [], 'crowd_level': [],'time_end': [], 'people_count': []}
        result_queue = {'time': [], 'queue': [], 'queue_count': [], 'wait_time': []}
        id_frame = 0
        boxes = {}
        frames = {}
        people_counts = {}
        reid_boxes = {}
        queue_wait = {}
        t1 = time.time()

        while state.process:
            
            id_frame += 1
            state.process = False
            #initialize dict to store the results

            for video,cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                state.process = True
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB)
                for test_ind in range(100):
                    if video not in boxes or id_frame % skip_frame == 0:
                        #process the frame
                        # print('process', id_frame)
                        boxes__ = []
                        for area_name, area in areas_info[video].items():
                            #draw rectangle area on frame
                            x1, y1, x2, y2 = list(map(int, area))
                            #create new frame with area
                            frame_area = frame[y1:y2, x1:x2]
                            #detect object in frame
                            boxes_ = detect(model,frame_area)
                            people_count = count_base_motion(motion_detectors[video],frame_area,boxes_)
                            people_counts[area_name] = people_count
                            x1, y1, x2, y2 = list(map(int, area))
                            boxes_[:, 0] += x1
                            boxes_[:, 1] += y1
                            boxes_[:, 2] += x1
                            boxes_[:, 3] += y1
                            boxes__.append(boxes_)

                        boxes[video] = np.concatenate(boxes__)
                            
                    #draw the boxes
                    for queue_name, queue in queues_info[video].items():
                        if queue_name not in queue_optical_flow:
                            queue_optical_flow[queue_name] = OpticalFlow(frame=frame, queue=queue)
                        distance_change = queue_optical_flow[queue_name].get_speed(frame) // 1.3 + 0.1
                        if distance_change < 0.2:
                            distance_change = 0.001
                        # print('distance_change', distance_change)
                        queue_distance[queue_name] = distance_change 
                    centers = np.zeros((1,2))
                    centers = (boxes[video][:, 2:] + boxes[video][:, :2]) / 2
                    

                    for area_name, area in areas_info[video].items():
                        
                        # count number of center in each area
                        count = 0
                        count = ((centers[:, 0] > area[0]) & (centers[:, 0] < area[2]) & (centers[:, 1] > area[1]) & (centers[:, 1] < area[3])).sum()
                        # check crowd level in area
                        crowd_range = areas_crowd[video][area_name]
                        if count < crowd_range['medium']:
                            crowd_level = '1low'
                        elif count < crowd_range['high']:
                            crowd_level = '2medium'
                        elif count < crowd_range['very_high']:
                            crowd_level = '3high'
                        else:
                            crowd_level = '4critical'
                        
                        for queue_name in queue_in_area[area_name]:
                            # count number of people in queue
                            queue = queues_info[video][queue_name]
                            queue_count = count_people_in_queue(frame,queue,centers)
                            result_queue['queue_count'].append(queue_count)
                            result_queue['queue'].append(queue_name)
                            # calculate wait time
                            e_wait = 0.3
                            wait_time = queue_count / ( e_wait * queue_distance[queue_name])
                            # print('wait_time', wait_time)
                            if wait_time > 1000.0:
                                wait_time =  fps[video]  * 30
                            if queue_name not in queue_wait:
                                queue_wait[queue_name] = wait_time
                            else:
                                queue_wait[queue_name] = 0.9* queue_wait[queue_name]  + 0.1* wait_time 
                            result_queue['wait_time'].append(queue_wait[queue_name] / fps[video])
                            result_queue['time'].append(timestamp2datetime( id_frame / fps[video] + 1662656400))
                        #update results dict
                        # fps[video] = 30
                        results['time'].append(timestamp2datetime( id_frame / fps[video] + 1662656400))
                        results['time_end'].append(timestamp2datetime( id_frame / fps[video] + 1 + 1662656400))
                        results['area'].append(area_name)
                        results['count'].append(count)
                        results['crowd_level'].append(crowd_level)
                        results['people_count'].append(people_counts[area_name])
                    frame = show_settings.show(frame, boxes[video], lines_info[video],queues_info[video],areas_info[video],stroke_color,stroke_width)
                    frames[video] = frame
            # show_results.show(results,result_queue,frames,placeholder,areas_info,queue_in_area)
        t2 = time.time()
        print('time', t2 - t1)
        print('frame', id_frame)

if __name__ == "__main__":

    main()