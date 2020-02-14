import cv2
from facenet_pytorch import MTCNN
from PIL import Image

def boost_contrast(imgs, alpha):
    return [cv2.convertScaleAbs(img , alpha=alpha, beta=0.0) for img in imgs]

# From: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
def bb_iou (boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

class FaceDetector:
    def __init__(self, min_face_size, margin, prob_threshold):
        self.detector = MTCNN(keep_all=True, post_process=False, device='cuda:0', select_largest=False, min_face_size=min_face_size)
        self.margin = margin
        self.prob_threshold = prob_threshold
        
    def crop_face(self, img, box):
        max_x = img.shape[1] - 1
        max_y = img.shape[0] - 1
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        x1 = max([x1 - self.margin, 0])
        y1 = max([y1 - self.margin, 0])
        x2 = min([x2 + self.margin, max_x])
        y2 = min([y2 + self.margin, max_y])
        face = img[y1:y2, x1:x2]
        new_box = [[x1, y1], [x2, y2]]
        
        return face, new_box

    def filter_faces(self, frame_data):
        faces = {}

        for frame_datum in frame_data:
            for face_data in frame_datum['faces']:
                face_id = face_data['id']
                
                if face_id in faces:
                    faces[face_id].append(face_data['prob'])
                else:
                    faces[face_id] = [face_data['prob']]

        num_frames = len(frame_data)
        face_ids_to_del = set()
        avg_probs = {}

        for face_id, probs in faces.items():
            if len(probs) < (num_frames / 2):
                print(f'FaceDetector::filter_faces: Face with id {face_id} failed to appear in >= {num_frames / 2} frames.')
                face_ids_to_del.add(face_id)
            avg_probs[face_id] = sum(probs) / len(probs)

        faces_remaining = len(faces) - len(face_ids_to_del)

        if faces_remaining > 2:
            print('FaceDetector::filter_faces: More than 2 faces. Only keeping the two with the highest avg prob.')
            avg_probs_sorted = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)

            for face_id, avg_prob in avg_probs_sorted[2:]:
                face_ids_to_del.add(face_id)

        filtered_frame_data = []
        for frame_datum in frame_data:
            faces_filtered = []

            for face_data in frame_datum['faces']:
                if face_data['id'] in face_ids_to_del:
                    continue
                else:
                    faces_filtered.append(face_data)

            frame_datum['faces'] = faces_filtered
            filtered_frame_data.append(frame_datum)

        return filtered_frame_data
    
    def detect(self, video, num_frames, filt=True):
        vc = cv2.VideoCapture(video)        
        imgs = []
        
        for i in range(num_frames):
            success, img = vc.read()
            if success:
                imgs.append(img)
            else:
                print('FaceDetector::detect: cv2::VideoCapture::read call failed.')
                return []
        
        imgs_pil = [Image.fromarray(i) for i in imgs]
        video_boxes, video_probs, video_landmarks = self.detector.detect(imgs_pil, landmarks=True)
        past_faces = []
        frame_data = []
        face_id = 0
        
        def search_past_faces(box, iou_threshold):
            i = 0
            for past_face in past_faces:
                iou = bb_iou(past_face['box'], face_box)
                if iou > iou_threshold:
                    return (True, i)
                else:
                    i += 1
            return (False, i)
            
        
        for frame, (frame_boxes, frame_probs, frame_landmarks) in enumerate(zip(video_boxes, video_probs, video_landmarks)):
            contrast_boosted = False
            img = imgs[frame]

            if frame_boxes is None:
#                 print(f'FaceDetector::detect: No faces in frame {frame}. Boosting contrast.')
                img = boost_contrast([img], 3.0)[0]
                img_pil = Image.fromarray(img)
                frame_boxes, frame_probs, frame_landmarks = self.detector.detect(img_pil, landmarks=True)
                contrast_boosted = True
                
                if frame_boxes is None:
#                     print(f'FaceDetector::detect: No faces after contrast boost. Proceeding to next frame.')
                    frame_data.append({'frame': frame, 'faces': [], 'contrast_boosted': contrast_boosted})
                    continue

            face_data = []
            
            for face_box, face_prob, face_landmarks in zip(frame_boxes, frame_probs, frame_landmarks):
                if face_prob < self.prob_threshold:
                    continue
               
                face_cropped, new_box = self.crop_face(img, face_box)
                found, idx = search_past_faces(face_box, 0.5)
                
                if found:
                    past_faces[idx]['box'] = face_box
                    face_data.append({'id': past_faces[idx]['id'], 'box': new_box, 'prob': face_prob, 'landmarks': face_landmarks, 'img': face_cropped})
                else:
                    past_faces.append({'box': face_box , 'id': face_id})
                    face_data.append({'id': face_id, 'box': new_box, 'prob': face_prob, 'landmarks': face_landmarks, 'img': face_cropped})
                    face_id += 1

            frame_data.append({'frame': frame, 'faces': face_data, 'contrast_boosted': contrast_boosted})

        return frame_data if not filt else self.filter_faces(frame_data) 
                