import cv2
import numpy as np


class ContrastPreprocessor:
    def __init__(self, alpha, beta, adaptive=False, threshold=0):
        self.alpha = alpha
        self.beta = beta
        self.adaptive = adaptive
        self.threshold = threshold

    def preprocess(self, imgs):
        preprocessed_imgs = []

        if self.adaptive:
            for img in imgs:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray_img)

                if brightness < self.threshold:
                    preprocessed_imgs.append(
                        cv2.convertScaleAbs(img,
                                            alpha=self.alpha,
                                            beta=self.beta))
                else:
                    preprocessed_imgs.append(img)
        else:
            preprocessed_imgs = [
                cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
                for img in imgs
            ]

        return preprocessed_imgs
