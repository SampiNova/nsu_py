import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backend_bases as plt_bb
import cv2
import os
import random

cwd = "/".join(os.getcwd().split('/')[:-1])

images_path = f"{cwd}/lesson6/nails_segmentation/images"
labels_path = f"{cwd}/lesson6/nails_segmentation/labels"
pairs = []
N = 0


def read_pairs():
    global pairs, N
    for image_name in os.listdir(images_path):
        image = cv2.imread(images_path + "/" + image_name)[:, :, ::-1]
        label = cv2.imread(labels_path + "/" + image_name)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        pairs += [(image, label)]
    N = len(pairs)


def augmentation(data, mode):
    img, labl = data
    width, height, _ = img.shape
    if mode == 0:
        m = random.choice([0, 1])
        if m:
            return [img[::-1], labl[::-1]]
        else:
            return [img[:][::-1], labl[:][::-1]]
    elif mode == 1:
        rot_mat = cv2.getRotationMatrix2D((width // 2, height // 2), np.random.randint(1, 360), 1.0)
        rimg = cv2.warpAffine(img, rot_mat, (256, 256), flags=cv2.INTER_LINEAR)
        rlabl = cv2.warpAffine(labl, rot_mat, (256, 256), flags=cv2.INTER_LINEAR)
        return [rimg, rlabl]
    elif mode == 2:
        x, y = random.randint(0, width - 20), random.randint(0, height - 20)
        w, h = random.randint(width - x, width), random.randint(height - y, height)
        return [img[y:y + h][x:x + w], labl[y:y + h][x:x + w]]
    else:
        k = random.randint(6, 25)
        return [cv2.blur(img, (k, k)), cv2.blur(labl, (k, k))]


def gen_nails(count):
    global pairs, N
    if count > N:
        return None
    idx = 0
    pairs_ = pairs.copy()
    while True:
        np.random.shuffle(pairs_)
        if idx + count > N:
            yield pairs_[idx:] + pairs_[:(idx + count) % N]
        else:
            yield pairs_[idx: idx + count]
            idx += count


def key_press_event(event: plt_bb.KeyEvent):
    global axis1, axis2, idx, res
    if event.key == "right":
        idx += 1
    elif event.key == "left":
        idx -= 1
    else:
        return
    if idx >= len(res):
        idx = 0
    if idx < 0:
        idx = len(res) - 1

    axis1.imshow(res[idx][0])
    axis2.imshow(res[idx][1], cmap="gray")
    axis1.figure.canvas.draw()


idx = 0
if __name__ == "__main__":
    read_pairs()
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2)

    g = gen_nails(20)
    D = list(next(g))

    res = []
    maxh, maxw = 0, 0
    for obj in D:
        m = np.random.randint(0, 4)
        res.append(augmentation(obj, m))
        if res[-1][0].shape[0] > maxw:
             maxw = res[-1][0].shape[0]
        if res[-1][0].shape[1] > maxh:
            maxh = res[-1][1].shape[1]
    for i in range(len(res)):
        res[i][0] = cv2.resize(res[i][0], (maxw, maxh))
        res[i][1] = cv2.resize(res[i][1], (maxw, maxh))

    axis1.imshow(res[idx][0])
    axis2.imshow(res[idx][1], cmap="gray")

    key_press_event_id = fig.canvas.mpl_connect("key_press_event", key_press_event)
    plt.show()
