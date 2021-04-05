import os, random, copy, math
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt


BASE_IMAGE_SIZE = 512

goal_image = cv2.resize(cv2.imread('ryuk.png'), (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE))
edges_image = cv2.Canny(goal_image, 100, 200, apertureSize=3, L2gradient=True)


def draw_images(*images):
    for image in images:
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def main():
    draw_images(goal_image, edges_image)


if __name__ == "__main__":
    main()