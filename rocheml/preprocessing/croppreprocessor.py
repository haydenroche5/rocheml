class CropPreprocessor:
    def __init__(self, box_origin, box_width, box_height):
        self.box_origin = box_origin
        self.box_width = box_width
        self.box_height = box_height

    def preprocess(self, imgs):
        crops = []

        for img in imgs:
            crop = img[self.box_origin[1]:self.box_origin[1] +
                       self.box_height, self.box_origin[0]:self.box_origin[0] +
                       self.box_width]
            crops.append(crop)

        return crops
