import face_recognition
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Load the jpg file into a numpy array
img = cv2.imread('satya.jpg')
size = img.shape

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(img)[0]
inner_face_pts = []
inner_face_pts.extend(face_landmarks_list['left_eyebrow'])
inner_face_pts.extend(face_landmarks_list['right_eyebrow'])
inner_face_pts.extend(face_landmarks_list['nose_bridge'])
inner_face_pts.extend(face_landmarks_list['nose_tip'])
inner_face_pts.append(face_landmarks_list['top_lip'][0])
inner_face_pts.append(face_landmarks_list['top_lip'][7])
outer_face_pts = face_landmarks_list['chin'][2:15]
face_pts = np.array(outer_face_pts + inner_face_pts, dtype='double')

# Based on http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp.
model_left_eyebrow_pts = [
    (-6.825897, 6.760612, 4.402142),
    (-6.137002, 7.271266, 5.200823),
    (-4.861131, 7.878672, 6.601275),
    (-2.533424, 7.878085, 7.451034),
    (-1.330353, 7.122144, 6.903745),
]
model_right_eyebrow_pts = [(1.330353, 7.122144, 6.903745),
                           (2.533424, 7.878085, 7.451034),
                           (4.861131, 7.878672, 6.601275),
                           (6.137002, 7.271266, 5.200823),
                           (6.825897, 6.760612, 4.402142)]
model_nose_bridge_pts = [(0, 5.862829, 7.65405), (0, 4.547349, 6.66534633),
                         (0, 3.231869, 5.67664267), (0, 1.916389, 4.687939)]
model_nose_tip_pts = [(-1.930245, 0.424351, 5.914376),
                      (-0.746313, 0.348381, 6.263227),
                      (0.000000, 0.000000, 6.763430),
                      (0.746313, 0.348381, 6.263227),
                      (1.930245, 0.424351, 5.914376)]
model_left_lip_corner_pt = (-2.774015, -2.080775, 5.048531)
model_right_lip_corner_pt = (2.774015, -2.080775, 5.048531)
model_chin_pts = [(-7.308957, 0.913869, 0.000000),
                  (-6.775290, -0.730814, -0.012799),
                  (-5.665918, -3.286078, 1.022951),
                  (-5.011779, -4.876396, 1.047961),
                  (-4.056931, -5.947019, 1.636229),
                  (-1.833492, -7.056977, 4.061275),
                  (0.000000, -7.415691, 4.070434),
                  (1.833492, -7.056977, 4.061275),
                  (4.056931, -5.947019, 1.636229),
                  (5.011779, -4.876396, 1.047961),
                  (5.665918, -3.286078, 1.022951),
                  (6.775290, -0.730814, -0.012799),
                  (7.308957, 0.913869, 0.000000)]

model_inner_face_pts = []
model_inner_face_pts.extend(model_left_eyebrow_pts)
model_inner_face_pts.extend(model_right_eyebrow_pts)
model_inner_face_pts.extend(model_nose_bridge_pts)
model_inner_face_pts.extend(model_nose_tip_pts)
model_inner_face_pts.append(model_left_lip_corner_pt)
model_inner_face_pts.append(model_right_lip_corner_pt)
model_outer_face_pts = model_chin_pts
# model_face_pts = np.array(model_outer_face_pts + model_inner_face_pts,
#                           dtype='double')
combined_list = model_outer_face_pts + model_inner_face_pts
new_list = []
for pt in combined_list:
    new_list.append((pt[0], pt[1], pt[2] - 6.763430))
model_face_pts = np.array(new_list, dtype='double')
# x = [point[0] for point in new_list]
# y = [point[1] for point in new_list]
# z = [point[2] for point in new_list]
# plt.plot(x, y, 'ro', alpha=0.5)
# for i in range(len(x)):
#     plt.text(x[i], y[i], str(i))
# plt.savefig('landmark_locations.png')

# Camera internals
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype='double')

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
(success, rotation_vector,
 translation_vector) = cv2.solvePnP(model_face_pts,
                                    face_pts,
                                    camera_matrix,
                                    dist_coeffs,
                                    flags=cv2.SOLVEPNP_ITERATIVE)

print(f'Rotation Vector:\n {rotation_vector}')
print(f'Translation Vector:\n {translation_vector}')

(nose_end_point2D,
 jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                               translation_vector, camera_matrix, dist_coeffs)

for i, p in enumerate(face_pts):
    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    cv2.putText(img, str(i), (int(p[0]), int(p[1]) - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

p1 = (int(face_landmarks_list['nose_tip'][2][0]),
      int(face_landmarks_list['nose_tip'][2][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(img, p1, p2, (255, 0, 0), 2)

# Display image
cv2.imwrite('output.jpg', img)
