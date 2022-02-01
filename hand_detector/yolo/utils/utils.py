import cv2
import numpy as np
from hand_detector.yolo.preprocess.yolo_flag import Flag

f = Flag()
grid = f.grid
grid_size = f.grid_size
alpha = f.alpha


def draw_grid(image, bbox):
    for i in range(0, grid + 1):
        image = cv2.line(image, (0, i * grid_size), (grid * grid_size, i * grid_size), f.line_color, 2)
        image = cv2.line(image, (i * grid_size, 0), (i * grid_size, grid * grid_size), f.line_color, 2)

    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    image = cv2.circle(image, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
    i, j = int(np.floor(center[0] / grid_size)), int(np.floor(center[1] / grid_size))
    glassy_image = image.copy()
    i, j = int(i), int(j)
    x = i * grid_size
    y = j * grid_size
    glassy_image = cv2.rectangle(glassy_image, (x, y), (x + grid_size, y + grid_size), f.grid_color, -1)
    image = cv2.addWeighted(glassy_image, alpha, image, 1 - alpha, 0)
    return image


def visualize(image, yolo_out=None, title='output', RGB2BGR=False):
    if yolo_out is not None:
        predicting_boxes = yolo_out[:, :, 0]
        i, j = np.squeeze(np.where(predicting_boxes == np.amax(predicting_boxes)))

        if predicting_boxes[i, j] >= f.threshold:
            bbox = yolo_out[i, j, 1:] * f.target_size
            image = draw_grid(image, bbox)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            image = cv2.rectangle(image, (x1, y1), (x2, y2), f.box_color, 2)

    if RGB2BGR:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow(title, image)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
