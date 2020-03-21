# Taken from Deep Learning for Computer Vision with Python, Starter Bundle

from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, imgs):
        return [
            img_to_array(img, data_format=self.data_format) for img in imgs
        ]
