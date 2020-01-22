import cv2
import os
import numpy as np


class FaceExtractor:
    def __init__(self, prototxt_file_path, model_file_path):
        if not os.path.exists(prototxt_file_path):
            raise Exception(
                'FaceExtractor::__init__: Prototxt file {} does not exist.'.
                format(prototxt_file_path))

        if not os.path.exists(model_file_path):
            raise Exception(
                'FaceExtractor::__init__: Model file {} does not exist.'.
                format(model_file_path))

        self.face_detection_model = cv2.dnn.readNetFromCaffe(
            prototxt_file_path, model_file_path)

    def extract_from_img(self, img):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300))
        self.face_detection_model.setInput(blob)
        detections = self.face_detection_model.forward()
        confidence = detections[0, 0, 0, 2]
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        face = img[startY:endY, startX:endX]

        return (confidence, face)
