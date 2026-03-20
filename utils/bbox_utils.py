def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox

    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    x1,y1,x2,y2 = bbox
    return bbox[2] - bbox[0]


def get_bbox_height(bbox):
    x1, y1, x2, y2 = bbox
    return bbox[3] - bbox[1]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def calculate_bbox_area(bbox):
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return width * height


def calculate_overlap_ratio(inner_bbox, outer_bbox):
    inner_area = calculate_bbox_area(inner_bbox)
    if inner_area == 0:
        return 0.0

    x_left = max(float(inner_bbox[0]), float(outer_bbox[0]))
    y_top = max(float(inner_bbox[1]), float(outer_bbox[1]))
    x_right = min(float(inner_bbox[2]), float(outer_bbox[2]))
    y_bottom = min(float(inner_bbox[3]), float(outer_bbox[3]))

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    return float(overlap_area / inner_area)


def point_to_bbox_distance(point, bbox):
    point_x, point_y = point
    x1, y1, x2, y2 = bbox

    clamped_x = min(max(float(point_x), float(x1)), float(x2))
    clamped_y = min(max(float(point_y), float(y1)), float(y2))

    dx = float(point_x) - clamped_x
    dy = float(point_y) - clamped_y
    return float((dx ** 2 + dy ** 2) ** 0.5)
