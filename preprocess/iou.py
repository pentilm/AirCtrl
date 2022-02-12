def iou(grid_box, obj_box):
    grid_area = (grid_box[2] - grid_box[0]) * (grid_box[3] - grid_box[1])

    x1 = max(grid_box[0], obj_box[0])
    y1 = max(grid_box[1], obj_box[1])
    x2 = min(grid_box[2], obj_box[2])
    y2 = min(grid_box[3], obj_box[3])

    if (y2 - y1) < 0 != (x2 - x1) < 0:
        intersection = 0
    else:
        intersection = (y2 - y1) * (x2 - x1)

    intersection_over_union = intersection / grid_area
    return intersection_over_union
