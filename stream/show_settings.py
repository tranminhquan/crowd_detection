import numpy as np
import cv2

def show(frame, boxes, lines_info, queues_info, areas_info,stroke_color, stroke_width):
    centers = np.zeros((1,2))
    centers = (boxes[:, 2:] + boxes[:, :2]) / 2
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x,y) in centers:
        x, y = list(map(int, [x, y]))
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)

    # get center from xyxy of boxes calculate by numpy
    # initialize emty center numpy array with shape (1,2)
    # draw lines_info to frame
    for line_name, line in lines_info.items():
        x1, y1, x2, y2 = list(map(int, line))
        cv2.line(frame, (x1, y1), (x2, y2), stroke_color, stroke_width)
        # cv2.putText(frame, line_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stroke_color, stroke_width)
    for queue_name, queue in queues_info.items():
        for p_index in range(len(queue) - 1):
            cv2.line(frame,(queue[p_index][0],queue[p_index][1]),(queue[p_index+1][0],queue[p_index+1][1]),stroke_color,stroke_width)
        cv2.line(frame,(queue[-1][0],queue[-1][1]),(queue[0][0],queue[0][1]),stroke_color,stroke_width)

    for area_name, area in areas_info.items():
        
        #draw rectangle area on frame
        x1, y1, x2, y2 = list(map(int, area))
        cv2.rectangle(frame, (x1, y1), (x2, y2), stroke_color, stroke_width)
        #put text on frame
        cv2.putText(frame, area_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stroke_color, stroke_width)
    return frame