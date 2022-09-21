import os
import streamlit as st
import cv2
from stream import show_sidebar, show_drawing
import numpy as np
import plotly.express as px  
import plotly.figure_factory as ff
import pandas as pd 
from threading import Thread
from detect import detect
import torch
ROOT_PATH = os.path.join(os.getcwd(), 'videos')

def get_images(video_path):
    #read first frame of video with cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame

def main():

    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.classes = [0]

    # state = SessionState.get(
    #     upload_key=None, enabled=True, start=False, run=False)
    state = st.session_state
    st.write("### Choose videos")
    path = st.multiselect("", os.listdir(ROOT_PATH))
    path = [os.path.join(ROOT_PATH, p) for p in path]

    drawing_mode, stroke_width, stroke_color, skip_frame, runtime_type = show_sidebar.show()
    print('stroke color', stroke_color)
    if runtime_type == 'GPU':
        model.cuda()
    else:
        model.cpu()

    images = []
    areas_info = {}
    areas_draw = {}
    areas_crowd = {}
    for video_path in path:
        image = get_images(video_path)
        # Show the image with streamlit canvas
        image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB) 
        scale_width = image.shape[1] / 640
        scale_height = image.shape[0] / 480
        images.append(image)

        areas_info[video_path], areas_draw[video_path],areas_crowd[video_path] = show_drawing.show(stroke_width, stroke_color, image, drawing_mode, scale_width, scale_height, None, key = video_path)
    print('skip_frame', skip_frame)
    if st.button("Start setting"):
     
        state.start = True
        state.run = False
        caps = {}
        fps = {}
        for video_path in path:
            if video_path not in caps:
                caps[video_path] = cv2.VideoCapture(video_path)
                fps[video_path] = caps[video_path].get(cv2.CAP_PROP_FPS)
        state.caps = caps
        state.process = True
        placeholder = st.empty()
        results = {'time': [],'area': [], 'count': [], 'crowd_level': []}
        id_frame = 0
        boxes = {}
        frames = {}
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
                
                if video not in boxes or id_frame % skip_frame == 0:
                    #process the frame
                    # print('process', id_frame)
                    boxes[video] = detect(model,frame)

                #draw the boxes
                centers = np.zeros((1,2))
                centers = (boxes[video][:, 2:] + boxes[video][:, :2]) / 2
                for (x1, y1, x2, y2) in boxes[video]:
                    x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for (x,y) in centers:
                    x, y = list(map(int, [x, y]))
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)
                
                # get center from xyxy of boxes calculate by numpy
                # initialize emty center numpy array with shape (1,2)
                for area_name, area in areas_info[video].items():
                    
                    #draw rectangle area on frame
                    x1, y1, x2, y2 = list(map(int, area))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), stroke_color, stroke_width)
                    #put text on frame
                    cv2.putText(frame, area_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stroke_color, stroke_width)


                    # count number of center in each area
                    count = 0
                    count = ((centers[:, 0] > area[0]) & (centers[:, 0] < area[2]) & (centers[:, 1] > area[1]) & (centers[:, 1] < area[3])).sum()
                    # check crowd level in area
                    crowd_range = areas_crowd[video][area_name]
                    if count < crowd_range['medium']:
                        crowd_level = 'low'
                    elif count < crowd_range['high']:
                        crowd_level = 'medium'
                    elif count < crowd_range['very_high']:
                        crowd_level = 'high'
                    else:
                        crowd_level = 'very_high'
                        
                    #update results dict
                    # results['time'].append(id_frame / fps[video])
                    results['time'].append(id_frame)
                    results['area'].append(area_name)
                    results['count'].append(count)
                    results['crowd_level'].append(crowd_level)
                frames[video] = frame

            #convert results dict to pandas dataframe
            df = pd.DataFrame(results)
            #add column time_end = time + 1/fps to calculate the duration of each area
            df['time_end'] = df['time'] + 1
            print(df)
            with placeholder.container():
            
                #plot the results
                tab1, tab2 = st.tabs(['Count', 'Crowd level'])
                with tab1:
                    fig = px.line(df.iloc[-200:], x="time", y="count", color='area',line_shape="spline", render_mode="svg")
                    fig.update_yaxes(range=[0, 30])
                    st.write(fig)
                with tab2:
                    fig2 = px.timeline(df.iloc[-200:],x_start="time",x_end="time_end", y="area", color="crowd_level", color_discrete_map={'low': 'green', 'medium': 'yellow', 'high': 'orange', 'very_high': 'red'})
                    fig2.update_xaxes(tickformat = '%S')
                    st.write(fig2)
                for video, frame in frames.items():
                    st.image(frame, caption=video, width=640)

if __name__ == "__main__":
    main()