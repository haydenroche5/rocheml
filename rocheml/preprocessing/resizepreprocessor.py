class ResizePreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, imgs):
        return [img.resize((self.width, self.height)) for img in imgs]
