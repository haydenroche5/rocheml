import face_recognition
import numpy as np
import cv2


class FacePoseExtractor:
    def __init__(self):
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
        self.model_outer_face_pts = np.array(model_outer_face_pts,
                                             dtype='double')
        self.model_inner_face_pts = np.array(model_inner_face_pts,
                                             dtype='double')
        self.model_whole_face_pts = np.array(model_outer_face_pts +
                                             model_inner_face_pts,
                                             dtype='double')

    def get_face_poses(self, img):
        camera_matrix = self.get_camera_matrix(img)
        face_landmarks_list = face_recognition.face_landmarks(img)
        whole_face_pts_list = []
        inner_face_pts_list = []
        nose_pts_list = []
        for face_landmarks in face_landmarks_list:
            outer_face_pts = face_landmarks['chin'][2:15]
            inner_face_pts = []
            inner_face_pts.extend(face_landmarks['left_eyebrow'])
            inner_face_pts.extend(face_landmarks['right_eyebrow'])
            inner_face_pts.extend(face_landmarks['nose_bridge'])
            inner_face_pts.extend(face_landmarks['nose_tip'])
            inner_face_pts.append(face_landmarks['top_lip'][0])
            inner_face_pts.append(face_landmarks['top_lip'][7])
            whole_face_pts_list.append(
                np.array(outer_face_pts + inner_face_pts, dtype='double'))
            inner_face_pts_list.append(np.array(inner_face_pts,
                                                dtype='double'))
            nose_pts_list.append(face_landmarks['nose_bridge'][3])

        poses = []
        for whole_face_pts, inner_face_pts, nose_pt in zip(
                whole_face_pts_list, inner_face_pts_list, nose_pts_list):
            (_, whole_face_rotation_vector,
             whole_face_translation_vector) = cv2.solvePnP(
                 self.model_whole_face_pts,
                 whole_face_pts,
                 camera_matrix,
                 np.zeros((4, 1)),
                 flags=cv2.SOLVEPNP_ITERATIVE)
            (_, inner_face_rotation_vector,
             inner_face_translation_vector) = cv2.solvePnP(
                 self.model_inner_face_pts,
                 inner_face_pts,
                 camera_matrix,
                 np.zeros((4, 1)),
                 flags=cv2.SOLVEPNP_ITERATIVE)

            poses.append({
                'whole': {
                    'rotation': whole_face_rotation_vector,
                    'translation': whole_face_translation_vector
                },
                'inner': {
                    'rotation': inner_face_rotation_vector,
                    'translation': inner_face_translation_vector
                },
                'nose_point': model_nose_bridge_pts
            })

        return poses

    def get_camera_matrix(self, img):
        height = img.shape[0]
        width = img.shape[1]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]], [0, 0, 1]],
                                 dtype='double')

        return camera_matrix

    def draw_pose_lines(self, img):
        poses = self.get_face_poses(img)
        camera_matrix = self.get_camera_matrix(img)
        for pose in poses:
            (nose_end_point2D,
             jacobian) = cv2.projectPoints(np.array([
                 (0.0, 0.0, 1000.0)
             ]), pose['whole']['rotation'], pose['whole']['translation'],
                                           camera_matrix, np.zeros((4, 1)))

            p1 = (int(pose['nose_point'][0]), int(pose['nose_point'][1]))
            p2 = (int(nose_end_point2D[0][0][0]),
                  int(nose_end_point2D[0][0][1]))

            cv2.line(img, p1, p2, (255, 0, 0), 2)

        return img