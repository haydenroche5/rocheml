class RescalePreprocessor:
    def __init__(self, divisor):
        self.divisor = divisor

    def preprocess(self, image):
        return image / (self.divisor * 1.0)
