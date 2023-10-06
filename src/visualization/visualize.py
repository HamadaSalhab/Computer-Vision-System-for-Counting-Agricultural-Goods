import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_image(image_path):
    """
    Show image
    :param image_path: path to image
    :return:
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def show_image_with_anns(image, anns):
    """
    Show image with annotations
    :param image: cv2 image
    :param anns: mask annotations
    :return:
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    if len(anns) != 0:
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
    plt.show()
