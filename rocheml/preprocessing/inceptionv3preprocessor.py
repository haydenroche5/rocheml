from tensorflow.keras.applications.inception_v3 import preprocess_input


class InceptionV3Preprocessor:
    def __init__(self):
        pass

    def preprocess(self, image):
        return preprocess_input(image)
