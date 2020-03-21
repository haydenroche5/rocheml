class RescalePreprocessor:
    def __init__(self, divisor):
        self.divisor = divisor

    def preprocess(self, imgs):
        return [img / (self.divisor * 1.0) for img in imgs]
