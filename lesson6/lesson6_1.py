import matplotlib.pyplot as plt
import matplotlib.backend_bases as plt_bb
import numpy as np
import cv2
import os


def key_press_event(event: plt_bb.KeyEvent):
    print(event.key)


images_path = "nails_segmentation\\images"
labels_path = "nails_segmentation\\labels"


def read_pairs():
    pairs = []
    for image_name in os.listdir(images_path):
        image = cv2.imread(images_path + "\\" + image_name)
        label = cv2.imread(labels_path + "\\" + image_name)
        pairs += [(image, label)]
    return pairs


idx = 0
if __name__ == "__main__":
    pairs_of_images = read_pairs()
    fig, (axis1, axis2) = plt.subplots(nrows=2)

    axis1.imshow(pairs_of_images[idx][0])
    axis1.grid()
    axis2.imshow(pairs_of_images[idx][1])
    axis2.grid()

    key_press_event_id = fig.canvas.mpl_connect("key_press_event", key_press_event)
    plt.show()

'''fig1, _ = plt.subplot(1, 2, 1)
plt.imshow(pairs[0][0])
fig2, _ = plt.subplot(1, 2, 2)
plt.imshow(pairs[0][1])

key_press_event_id_1 = fig1.canvas.mpl_connect("key_press_event", key_press_event)
key_press_event_id_2 = fig2.canvas.mpl_connect("key_press_event", key_press_event)

plt.show()
fig1.canvas.mpl_disconnect(key_press_event_id_1)
fig2.canvas.mpl_disconnect(key_press_event_id_2)'''
