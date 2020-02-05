import face_recognition
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Load the jpg file into a numpy array
img = cv2.imread('satya.jpg')
size = img.shape

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(img)[0]
outer_face_pts = face_landmarks_list['chin'][2:15]
inner_face_pts = []
inner_face_pts.extend(face_landmarks_list['left_eyebrow'])
inner_face_pts.extend(face_landmarks_list['right_eyebrow'])
inner_face_pts.extend(face_landmarks_list['nose_bridge'])
inner_face_pts.extend(face_landmarks_list['nose_tip'])
inner_face_pts.append(face_landmarks_list['top_lip'][0])
inner_face_pts.append(face_landmarks_list['top_lip'][7])
face_pts = np.array(outer_face_pts + inner_face_pts, dtype='double')

# Based on http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp.
model_chin_pts = [(-7.308957, 0.913869, -6.76343),
                  (-6.775290, -0.730814, -6.776229),
                  (-5.665918, -3.286078, -5.740479),
                  (-5.011779, -4.876396, -5.715469),
                  (-4.056931, -5.947019, -5.1272009999999995),
                  (-1.833492, -7.056977, -2.7021549999999994),
                  (0.000000, -7.415691, -2.692996),
                  (1.833492, -7.056977, -2.7021549999999994),
                  (4.056931, -5.947019, -5.1272009999999995),
                  (5.011779, -4.876396, -5.715469),
                  (5.665918, -3.286078, -5.740479),
                  (6.775290, -0.730814, -6.776229),
                  (7.308957, 0.913869, -6.76343)]
model_left_eyebrow_pts = [
    (-6.825897, 6.760612, -2.361287999999999),
    (-6.137002, 7.271266, -1.5626069999999999),
    (-4.861131, 7.878672, -0.16215499999999938),
    (-2.533424, 7.878085, 0.6876040000000003),
    (-1.330353, 7.122144, 0.1403150000000002),
]
model_right_eyebrow_pts = [(1.330353, 7.122144, 0.1403150000000002),
                           (2.533424, 7.878085, 0.6876040000000003),
                           (4.861131, 7.878672, -0.16215499999999938),
                           (6.137002, 7.271266, -1.5626069999999999),
                           (6.825897, 6.760612, -2.361287999999999)]
model_nose_bridge_pts = [(0, 5.862829, 0.8906200000000002),
                         (0, 4.547349, -0.0980836699999994),
                         (0, 3.231869, -1.08678733),
                         (0, 1.916389, -2.0754909999999995)]
model_nose_tip_pts = [(-1.930245, 0.424351, -0.8490539999999998),
                      (-0.746313, 0.348381, -0.500203),
                      (0.000000, 0.000000, 0.0),
                      (0.746313, 0.348381, -0.500203),
                      (1.930245, 0.424351, -0.8490539999999998)]
model_left_lip_corner_pt = (-2.774015, -2.080775, -1.714899)
model_right_lip_corner_pt = (2.774015, -2.080775, -1.714899)

model_inner_face_pts = []
model_inner_face_pts.extend(model_left_eyebrow_pts)
model_inner_face_pts.extend(model_right_eyebrow_pts)
model_inner_face_pts.extend(model_nose_bridge_pts)
model_inner_face_pts.extend(model_nose_tip_pts)
model_inner_face_pts.append(model_left_lip_corner_pt)
model_inner_face_pts.append(model_right_lip_corner_pt)
model_outer_face_pts = model_chin_pts
model_face_pts = np.array(model_outer_face_pts + model_inner_face_pts,
                          dtype='double')

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
