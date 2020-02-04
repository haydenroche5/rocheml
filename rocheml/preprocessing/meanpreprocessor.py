# import the necessary packages
import cv2


# Taken from the pyimagesearch Practitioner Bundle.
class MeanPreprocessor:
    def __init__(self, r_mean, g_mean, b_mean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, seq):
        seq_length = seq.shape[0]
        for i in range(seq_length):
            # split the image into its respective Red, Green, and Blue
            # channels
            (b, g, r) = cv2.split(seq[i].astype("float32"))

            # subtract the means for each channel
            r -= self.r_mean
            g -= self.g_mean
            b -= self.b_mean

            # merge the channels back together
            seq[i] = cv2.merge([b, g, r])

        return seq
