import cv2


class ContrastPreprocessor:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def preprocess(self, imgs):
        return [
            cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
            for img in imgs
        ]
