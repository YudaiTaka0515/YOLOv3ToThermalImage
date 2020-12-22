# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import matplotlib.pyplot as plt

import colorsys
import os
from timeit import default_timer as timer
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yoloV3.model import yolo_eval, yolo_body, tiny_yolo_body
from yoloV3.utils import letterbox_image
from yoloV3.resizeImage import *
import os
from keras.utils import multi_gpu_model
OUTPUT_PATH = os.path.join(r"C:\Users\takah\Desktop\GM", "video1.avi")


class YOLO(object):
    _defaults = {
        # "model_path": r'I:\3.Data\RespData\weights\10_22\COCO2.h5',
        "model_path": r'I:\3.Data\RespData\weights\12_21\trained_weights_final.h5',
        "anchors_path": r'G:\GUI4CalcRR\model_data\yolo_anchors.txt',
        "classes_path": r'G:\GUI4CalcRR\model_data\nosemouse_classes.txt',
        # "classes_path": r'G:\GUI4CalcRR\model_data\nose_classes.txt',
        "score": 0.25,
        "iou": 0.45,
        "model_image_size": (320, 320),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights_noseonly must be a .h5 file.'

        # Load model, or construct model and load weights_noseonly.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def calc_detection_score(self, _out_classes, _out_scores, _out_boxes):
        """戻り値 : 口のboxとscore, 鼻のboxとscore"""
        face_id, mouse_id, nose_id = 0, 1, 2

        # 顔のDetection結果を削除
        face_indexes = []
        for i in range(len(_out_classes)):
            if self.class_names[_out_classes[i]] == 'face':
                face_indexes.append(i)

        for face_index in reversed(face_indexes):
            _out_classes = np.delete(_out_classes, face_index)
            _out_scores = np.delete(_out_scores, face_index)
            _out_boxes = np.delete(_out_boxes, face_index, axis=0)

        # print(len(_out_classes))
        if len(_out_classes) == 0:
            return None, 0, None, 0

        # 最も確信度が大きいB-boxを取得
        conf_max_index = np.argmax(_out_scores)
        conf_max_class = self.class_names[_out_classes[conf_max_index]]
        conf_max_box = _out_boxes[conf_max_index]
        top, left, bottom, right = conf_max_box
        height = abs(top-bottom)
        conf_max_center = [(top+bottom)/2, (left+right)/2]
        # print(conf_max_class)

        # 確信度が低いクラスのほうのB-boxについて, 確信度と選択B-boxとの距離でスコアを算出
        scores = []
        detection_scores = []

        for i, c in enumerate(_out_classes):
            if self.class_names[c] != conf_max_class:
                top, left, bottom, right = _out_boxes[i]
                center = [(top + bottom) / 2, (left + right) / 2]
                detection_score = _out_scores[i]
                detection_scores.append(detection_scores)
                distance = np.sqrt((center[0]-conf_max_center[0])**2 + (center[1]-conf_max_center[1])**2)
                score = detection_score + np.tanh(distance/height/2)
                scores.append(score)
                # print(score, distance, detection_score)
            else:
                scores.append(0)

        if len(scores) == 0:
            another_score = 0
            another_box = None
        else:
            another_score = np.max(scores)
            another_box = _out_boxes[np.argmax(scores)]

        if conf_max_class == 'mouse':
            return conf_max_box, np.max(_out_scores), another_box, another_score
        else:
            return another_box, another_score, conf_max_box, np.max(_out_scores)

    def detect_face(self, pil_image, pre_box_n=[0, 0, 0, 0], pre_box_m=[0, 0, 0, 0], display_score=True, display_face=False, ):
        """
        :param pil_image: Imageオブジェクト(pillow)
        :param pre_box_n ; 前フレームの鼻のB-box
        :param pre_box_m ; 前フレームの口のB-box
        :param display_score : scoreを表示するか(T/F)
        :param display_face : faceを表示するか
        :return:
        """
        start = timer()
        font = ImageFont.truetype(font=r'font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * pil_image.size[1] + 0.5).astype('int32'))
        thickness = (pil_image.size[0] + pil_image.size[1]) // 300

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(pil_image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (pil_image.width - (pil_image.width % 32),
                              pil_image.height - (pil_image.height % 32))
            boxed_image = letterbox_image(pil_image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [pil_image.size[1], pil_image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        face_box = [0, 0, 0, 0]

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'face':
                continue
            box = out_boxes[i]
            face_score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(pil_image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(pil_image.size[0], np.floor(right + 0.5).astype('int32'))
            face_box = [top, left, bottom, right]

            # My kingdom for a good redistributable image drawing library.

        draw = ImageDraw.Draw(pil_image)

        if face_box != [0, 0, 0, 0]:
            top, left, bottom, right = face_box
            for i in range(thickness):
                draw.rectangle(
                                [left + i, top + i, right - i, bottom - i],
                                outline=(255, 0, 0))
            # scoreを非表示に
            label = '{} {:.2f}'.format('face', face_score)
            label_size = draw.textsize(label, font)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            if display_score:
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=(255, 0, 0))
                draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        end = timer()
        del draw
        return pil_image, np.array(face_box)

    def close_session(self):
        self.sess.close()

    def detect_distance(self, pil_image, display_score=False, display_face=False, ):
        """
        :param pil_image: Imageオブジェクト(pillow)
        :param pre_box_n ; 前フレームの鼻のB-box
        :param pre_box_m ; 前フレームの口のB-box
        :param display_score : scoreを表示するか(T/F)
        :param display_face : faceを表示するか
        :return:
        """
        start = timer()
        font = ImageFont.truetype(font=r'font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * pil_image.size[1] + 0.5).astype('int32'))
        thickness = (pil_image.size[0] + pil_image.size[1]) // 300

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(pil_image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (pil_image.width - (pil_image.width % 32),
                              pil_image.height - (pil_image.height % 32))
            boxed_image = letterbox_image(pil_image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [pil_image.size[1], pil_image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        mouse_box, mouse_score, nose_box, nose_score = self.calc_detection_score(out_classes, out_scores, out_boxes)

        draw = ImageDraw.Draw(pil_image)
        if mouse_box is not None:
            top, left, bottom, right = mouse_box
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(255, 0, 0))
            # scoreを非表示に
            label = '{} {:.2f}'.format('mouse', mouse_score)
            label_size = draw.textsize(label, font)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            if display_score:
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=(255, 0, 0))
                draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        else:
            mouse_box = np.array([0, 0, 0, 0])

        if nose_box is not None:
            top, left, bottom, right = nose_box
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(0, 0, 255))
            # scoreを非表示に
            label = '{} {:.2f}'.format('nose', nose_score)
            label_size = draw.textsize(label, font)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            if display_score:
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=(0, 0, 255))
                draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        else:
            nose_box = np.array([0, 0, 0, 0])

        end = timer()
        # print(end - start)
        del draw
        return pil_image, nose_box, mouse_box

    def detect_temp(self, pil_image, display_score=True, display_face=False):
        """
        :param pil_image: Imageオブジェクト(pillow)
        :param display_score : scoreを表示するか(T/F)
        :param display_face : faceを表示するか
        :return:
        """
        start = timer()
        font = ImageFont.truetype(font=r'G:\GUI4CalcRR\font\FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * pil_image.size[1] + 0.5).astype('int32'))
        thickness = (pil_image.size[0] + pil_image.size[1]) // 300

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(pil_image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (pil_image.width - (pil_image.width % 32),
                              pil_image.height - (pil_image.height % 32))
            boxed_image = letterbox_image(pil_image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [pil_image.size[1], pil_image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        nose_box = [0, 0, 0, 0]
        mouse_box = [0, 0, 0, 0]
        face_box = [0, 0, 0, 0]

        max_mouse_score = 0
        max_nose_score = 0
        max_face_score = 0

        count = 0

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            # 顔のDetectionは表示しない
            # if predicted_class == 'face' and not display_face:
            #     continue
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(pil_image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(pil_image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))
            draw = ImageDraw.Draw(pil_image)

            # My kingdom for a good redistributable image drawing library.
            if predicted_class == 'nose':
                count += 1
                for j in range(thickness):
                    draw.rectangle(
                        [left + j, top + j, right - j, bottom - j],
                        outline=(0, 0, 255))
                # scoreを非表示に
                label = '{} {:.2f}'.format('nose', score)
                label_size = draw.textsize(label, font)
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                if display_score:
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=(0, 0, 255))
                    draw.text(text_origin, label, fill=(255, 255, 255), font=font)
                nose_box = [top, left, bottom, right]

            # if predicted_class == 'mouse':
            #     count += 1
            #     for j in range(thickness):
            #         draw.rectangle(
            #             [left + j, top + j, right - j, bottom - j],
            #             outline=(255, 0, 0))
            #     # scoreを非表示に
            #     label = '{} {:.2f}'.format('mouse', score)
            #     label_size = draw.textsize(label, font)
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #
            #     if display_score:
            #         draw.rectangle(
            #             [tuple(text_origin), tuple(text_origin + label_size)],
            #             fill=(255, 0, 0))
            #         draw.text(text_origin, label, fill=(255, 255, 255), font=font)
            # #     mouse_box = [top, left, bottom, right]

        end = timer()
        # print(end - start)
        # print(count)
        del draw
        return pil_image, nose_box, mouse_box, face_box


