import cv2 as cv
import numpy as np
class OpticalFlow():
    def __init__(self,frame,queue,skip_frame = 1):
        super(OpticalFlow, self).__init__()
        self.skip_frame = skip_frame
        self.prev_frame = None
        self.flow = None
        self.flow_mask = None
        self.feature_params = dict(maxCorners = 100, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize = (5,5), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.frame_index = 0
        self.mask = self.create_mask(frame_shape = frame.shape[:2],queue = queue)
        self.current_track = []
        self.prev_track = []
        self.distances = 0

    def detect(self,frame):
        frame = cv.bitwise_and(frame,frame,mask=self.mask)
        # self.frame_index += 1
        if self.frame_index == 1:
            self.prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            return [],[]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        prev = cv.goodFeaturesToTrack(self.prev_gray, mask = None, **self.feature_params)
        next, status, error = cv.calcOpticalFlowPyrLK(self.prev_gray, gray, prev, None, **self.lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1].astype(int)
        # print('status',len(status))
        # good_old = prev.astype(int)
        # Selects good feature points for next positi1on
        good_new = next[status == 1].astype(int)
        # good_new = next.astype(int)
        self.prev_gray = gray.copy()
        # Updates previous good feature points

        return good_new,good_old

    
    def create_mask(self,frame_shape,queue):
        # fill frame with white
        frame_ = np.zeros(frame_shape, dtype = np.uint8)
        cv.fillPoly(frame_,pts = [np.array(queue)], color= (255,255,255))
        _, mask = cv.threshold(frame_, thresh=180, maxval=255, type=cv.THRESH_BINARY)
        # frame_ = cv.bitwise_and(frame, frame, mask=mask)
        return mask
        

    def get_speed(self,frame):
        self.frame_index +=1
        if self.frame_index == 1:
            good_new,good_old = self.detect(frame)
            return self.distances
        if self.frame_index % self.skip_frame == 0:
            good_new,good_old = self.detect(frame = frame)
            self.distances = np.mean(np.sqrt(np.sum((good_new - good_old)**2,axis=1)))
        return self.distances

        