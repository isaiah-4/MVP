import cv2
import sys
import numpy as np
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width
 

def draw_triangle(frame, bbox, color):
    y = int(bbox[3])
    x, _ = get_center_of_bbox(bbox)
    x = int(x)

    triangle_points = np.array([
        [x, y ],
        [x - 10, y - 20],
        [x + 10, y - 20]
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, color, 2)

    return frame


def draw_ellipse(frame, bbox, color, tracker_id=None):
    y2 = bbox[3]
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)


    cv2.ellipse(
        frame,
        center=(int(x_center), int(y2)),
        axes=(int(width), int(0.35 * width)),
        angle=0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    rectangle_Width = 40
    rectangle_Height=20
    x1_rect = x_center - rectangle_Width//2
    x2_rect = x_center + rectangle_Width//2
    y1_rect = (y2 -rectangle_Height//2) + 15
    y2_rect = (y2 + rectangle_Height//2) + 15

    if tracker_id is not None:
        tracker_text = str(tracker_id)
        text_size, _ = cv2.getTextSize(
            tracker_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )
        rectangle_Width = max(40, text_size[0] + 18)
        x1_rect = x_center - rectangle_Width // 2
        x2_rect = x_center + rectangle_Width // 2

        cv2.rectangle(
            frame,
            (int(x1_rect), int(y1_rect)), 
            (int(x2_rect), int(y2_rect)), 
            color,
            cv2.FILLED
            )

        x1_text = x_center - (text_size[0] // 2)

        cv2.putText(
            frame, 
            tracker_text,
            (int(x1_text), int(y2_rect - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            )
        
    return frame


def put_text_with_outline(
    frame,
    text,
    position,
    font_scale=0.5,
    color=(255, 255, 255),
    outline_color=(0, 0, 0),
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame
