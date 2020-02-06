from rocheml.cv.face.faceposeextractor import FacePoseExtractor
import sys
import cv2

img = cv2.imread('headshot_square.png')
fp_extractor = FacePoseExtractor()
img_w_lines = fp_extractor.draw_pose_lines(img)

cv2.imshow('', img_w_lines)
key = cv2.waitKey(0)
if key == ord('q'):
    sys.exit()
