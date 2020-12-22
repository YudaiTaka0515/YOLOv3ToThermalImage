import os
import glob
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pprint


# kerasV3用にconvertしたannotation_path
ANNOTATION_PATH = r"I:\3.Data\FaceDataset\KerasV3_1214_v2" \
                  r"\train\_annotations.txt"
TRAIN_DIR = r"I:\3.Data\FaceDataset\KerasV3_1214_v2\train"

new_annotation_path = "new_annotations.txt"

CLASSES = {"face": 0, "mouth": 1, "nose": 2}
COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
THICKNESS = 4


def thin_out_dataset():
    f = open(new_annotation_path, 'r')
    lines = f.readlines()
    print(len(lines))
    # return

    new_annotation_lines = []

    for i in range(len(lines)):
        line = lines[i].split(' ')
        img_name = line[0]
        img_path = os.path.join(TRAIN_DIR, img_name)

        pil_image = Image.fromarray(cv2.imread(img_path))
        draw = ImageDraw.Draw(pil_image)

        for j in range(1, len(line)):
            left, bottom, right, top, class_name = [int(temp) for temp in line[j].split(',')]
            for k in range(THICKNESS):
                draw.rectangle([left+k, top+k, right-k, bottom-k], outline=COLOR[class_name])

        del draw

        cv2.imshow('image', np.array(pil_image))
        k = cv2.waitKey(0)
        if k == ord('s'):
            new_annotation_lines.append(lines[i])
        elif k == ord('a'):

    pprint.pprint(new_annotation_lines)

    # with open(new_annotation_path, mode='w') as f:
    #     f.write(''.join(new_annotation_lines))


if __name__ == '__main__':
    thin_out_dataset()













