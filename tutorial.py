from cv.face.faceposeextractor import FacePoseExtractor
import sys

img = cv2.imread('headshot_square.png')
img_w_lines = fp_extractor.draw_pose_lines(img)

cv2.imshow('', img_w_lines)
key = cv2.waitKey(0)
if key == ord('q'):
    sys.exit()
