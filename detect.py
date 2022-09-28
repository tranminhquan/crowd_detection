from multiprocessing.pool import ThreadPool
import numpy as np
import cv2
import uuid
def detect(model, image):
    pool = ThreadPool(processes=10)
    async_result = pool.apply_async(model, (image,)) # tuple of args for foo
    result = async_result.get()
    boxes = result.pandas().xyxy[0]
    boxes = boxes.iloc[:,:4].values
    return boxes


def get_union_area(frame,areas):
    # convert frame to black and white
    frame = np.zeros(frame.shape[:2], dtype=np.uint8)
    # color areas of interest in frame with white
    for area in areas:
        # map area to int
        area = area.astype(int)
        frame[area[1]:area[3], area[0]:area[2]] = 255
    #count white pixels
    cv2.imwrite('frame.png',frame)
    count = np.count_nonzero(frame)
    return count 

def count_base_motion(motion_detector,frame,boxes):
    nBB = len(boxes)
    boxes_area = get_union_area(frame, boxes)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(src=frame, ksize=(5,5), sigmaX=0)
    fgmask = motion_detector.apply(frame)
    kernel = np.ones((5, 5))
    fgmask = cv2.dilate(fgmask, kernel, 1)
    cv2.imwrite('fgmask.jpg', fgmask)
    motion_count = np.count_nonzero(fgmask)
    # calculate number of people in area6
    people_count = int(nBB * motion_count / (boxes_area + 0.0001))
    return people_count

def predict_buz():
    
        # if not state.process:
        #         df = pd.DataFrame(results)
        #         predict = {'time': [],'area': [], 'count': []}
        #         predict_step = 100
        #         for video in path:
        #             for area in areas_info[video]:
        #                 print("start fitting")
        #                 # print(df[df['area'] == area][['count']])
        #                 SARIMAX_model = pm.auto_arima(df[df['area'] == area].iloc[:30][['count']].reset_index(drop = True), 
        #                         start_p=1, start_q=1,
        #                         test='adf',
        #                         max_p=3, max_q=3, m=12,
        #                         start_P=0, seasonal=True,
        #                         d=None, D=1, 
        #                         trace=False,
        #                         error_action='ignore',  
        #                         suppress_warnings=True, 
        #                         stepwise=True)
                        
        #                 print('start predicting' + area)
        #                 fit_data = SARIMAX_model.predict(n_periods=predict_step)
        #                 st.write(fit_data)
        #                 predict['time'].extend(list(map(lambda x: timestamp2datetime(x/fps[video]),list(range(predict_step)))))
        #                 predict['area'].extend([area] * predict_step)
        #                 predict['count'].extend(fit_data)
        #         predict = pd.DataFrame(predict)
        #         predict = predict.groupby(['area', 'time']).agg({'count': 'max'}).reset_index()
        #         # show predict result as line chart
        #         fig = px.line(predict, x="time", y="count", color='area',line_shape="linear", render_mode="svg")
        #         fig.update_yaxes(range=[0, 30])
        #         st.write(fig)
    pass


def reid_base_boxes(reid_boxes: dict,boxes):
    # calculate center of boxes
    centers = (boxes[:, 2:] + boxes[:, :2]) / 2
    expand = 15
    reid_boxes_ = {}
    # check if center of boxes is in reid_boxes
    for center in centers:
        have_id = False
        for id,reid_box in reid_boxes.items():
            if reid_box[0] < center[0] < reid_box[2] and reid_box[1] < center[1] < reid_box[3]:
                reid_boxes_[id] = [center[0]-expand,center[1]-expand,center[0]+expand,center[1]+expand]
                reid_boxes.pop(id)
                have_id = True
                break
        if not have_id:
            reid_boxes_[str(uuid.uuid1().int)] = [center[0]-expand,center[1]-expand,center[0]+expand,center[1]+expand]
    return reid_boxes_

def count_people_in_queue(frame,queue,centers):
    frame = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(frame,pts=[np.array(queue)],color=(255, 255, 255))
    # count number of cordination of centers is in queue
    count = 0
    for center in centers:
        if frame[int(center[1]),int(center[0])] == 255:
            count += 1
        cv2.circle(frame, (int(center[0]),int(center[1])), 5, (255, 255, 255), -1)
    cv2.imwrite('frame.png',frame)
    return count

def speed_track(frame,queue):
    # fill frame with white
    frame_ = np.zeros(frame.shape[:2], dtype = np.uint8)
    cv2.fillPoly(frame_,pts = [np.array(queue)], color= (255,255,255))
    _, mask = cv2.threshold(frame_, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    frame_ = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imwrite('mask.jpg',frame_)