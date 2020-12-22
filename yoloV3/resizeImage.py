from PIL import Image
import cv2
import numpy as np
black = [0x00, 0x00, 0x00]


def resize_square(img, size=(320, 320)):
    height, width, _ = img.shape
    if height < width:
        margin = (width-height) // 2
        reshaped_img = img[:, margin:-margin, :]
    else:
        margin = (height - width) // 2
        reshaped_img = img[margin:-margin, :, :]

    reshaped_img = cv2.resize(reshaped_img, dsize=size)

    return reshaped_img


def Expand2Square(np_img, background_color=(0, 0, 0)):
    height, width = np_img.shape[0], np_img.shape[1]
    if width == height:
        return np_img, width
    elif width > height:
        bottom = width-height
        result = cv2.copyMakeBorder(np_img, 0, bottom, 0, 0, cv2.BORDER_CONSTANT)
        return result, width
    else:
        right = height - width
        result = cv2.copyMakeBorder(np_img, 0, 0, 0, right, cv2.BORDER_CONSTANT)
        return result, height


def ResizeImage(np_img, shape=(320, 320)):
    image,  side_length = Expand2Square(np_img)
    # print(image.shape)
    # どのくらい大きくしたか
    scale = shape[0]/side_length
    resized_image = cv2.resize(image, shape)
    # print(resized_image.shape)

    return resized_image


def _Expand2Square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img, width
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img)
        return result, width
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img)
        return result, height


def _ResizeImage(pil_img, shape=(320, 320)):
    image,  side_length = Expand2Square(pil_img)
    resized_image = image.resize(shape)
    # どのくらい大きくしたか
    scale = shape[0]/side_length

    return resized_image, scale



