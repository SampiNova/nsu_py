import matplotlib.pyplot as plt
import matplotlib.backend_bases as plt_bb
import numpy as np
import cv2
import os


def key_press_event(event: plt_bb.KeyEvent):
    global axis11, axis12, axis21, axis22, idx, pairs_of_images
    if event.key == "right":
        idx += 1
    elif event.key == "left":
        idx -= 1
    else:
        return
    if idx >= len(pairs_of_images):
        idx = 0
    if idx < 0:
        idx = len(pairs_of_images) - 1

    axis11.imshow(pairs_of_images[idx][0])
    axis12.imshow(pairs_of_images[idx][2])
    axis21.imshow(pairs_of_images[idx][1], cmap="gray")
    axis22.imshow(pairs_of_images[idx][3], cmap="gray")

    axis11.figure.canvas.draw()


images_path = "nails_segmentation\\images"
labels_path = "nails_segmentation\\labels"


def read_pairs():
    pairs = []
    for image_name in os.listdir(images_path):
        image = cv2.imread(images_path + "\\" + image_name)[:, :, ::-1]
        label = cv2.imread(labels_path + "\\" + image_name)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(label, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cont_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
        cont_label = cv2.drawContours(np.zeros(label.shape), contours, -1, (255, 255, 255), 2)
        pairs += [(image, label, cont_image, cont_label)]
    return pairs


idx = 0
if __name__ == "__main__":
    pairs_of_images = read_pairs()
    fig, ((axis11, axis12), (axis21, axis22)) = plt.subplots(nrows=2, ncols=2)

    axis11.imshow(pairs_of_images[idx][0])
    axis12.imshow(pairs_of_images[idx][2])
    axis21.imshow(pairs_of_images[idx][1], cmap="gray")
    axis22.imshow(pairs_of_images[idx][3], cmap="gray")

    key_press_event_id = fig.canvas.mpl_connect("key_press_event", key_press_event)
    plt.show()
