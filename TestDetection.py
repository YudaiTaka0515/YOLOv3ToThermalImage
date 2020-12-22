from Detect import *
from ThinOutDataset import *


ANNOTATION_PATH = "new_annotations.txt"
TRAIN_DIR = r"I:\3.Data\FaceDataset\KerasV3_1214_v2\train"


CLASSES = {"face": 0, "mouth": 1, "nose": 2}
COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}


def test_detection(annotation_path, yolo):
    f = open(ANNOTATION_PATH, 'r')
    lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i].split(' ')
        img_name = line[0]
        img_path = os.path.join(TRAIN_DIR, img_name)

        pil_image = Image.fromarray(cv2.imread(img_path))
        draw = ImageDraw.Draw(pil_image)
        img = cv2.imread(img_path)

        detection_result, nose_box, mouth_box, face_box = yolo.detect(Image.fromarray(img))

        detection_result = np.array(detection_result)
        boxes = [face_box, mouth_box, nose_box]

        for j in range(1, len(line)):
            left, bottom, right, top, class_name = [int(temp) for temp in line[j].split(',')]
