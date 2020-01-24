class MeanSubPreprocessor:
    def __init__(self, means):
        self.means = means

    def preprocess(self, image):
        return image - self.means
