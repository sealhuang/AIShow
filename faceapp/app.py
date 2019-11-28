# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import time
import cv2 as cv
from PIL import Image
import streamlit as st

from faceinsight.detection import MTCNNDetector, show_bboxes
from faceinsight.detection.align_trans import get_reference_facial_points
from faceinsight.detection.align_trans import warp_and_crop_face

from recognizer import IdentityRecognizer
from recognizer import GenderRecognizer
from recognizer import AgeRecognizer
from recognizer import EmotionRecognizer
from facetracker import FaceTracker


#-- face crop apis
def get_square_crop_box(crop_box, box_scalar=1.0):
    """Get square crop box based on bounding box and the expanding scalar.
    Return square_crop_box and the square length.
    """
    center_w = int((crop_box[0] + crop_box[2]) / 2)
    center_h = int((crop_box[1] + crop_box[3]) / 2)
    w = crop_box[2] - crop_box[0]
    h = crop_box[3] - crop_box[1]
    box_len = max(w, h)
    delta = int(box_len * box_scalar / 2)
    square_crop_box = (center_w-delta, center_h-delta,
                       center_w+delta+1, center_h+delta+1)
    return square_crop_box, 2*delta+1

def crop_face(img, bounding_boxes, facial_landmarks, scalar, image_size,
              detect_multiple_faces=True):
    """Crop and align faces.

    Return:
        cropped faces: list of PIL.Images
    """
    # filter real faces based on detection confidence
    confidence_thresh = 0.85
    filtered_idx = bounding_boxes[:, 4]>=confidence_thresh
    filtered_bboxes = bounding_boxes[filtered_idx]
    filtered_facial_landmarks = facial_landmarks[filtered_idx]

    # if no faces found, return empty list
    if not len(filtered_bboxes):
        return []

    nrof_faces = len(filtered_bboxes)
    faces = []

    # detect multiple faces or not
    det = filtered_bboxes[:, 0:4]
    det_arr = []
    img_size = np.asarray(img.size)
    if nrof_faces>1:
        if detect_multiple_faces:
            for i in range(nrof_faces):
                det_arr.append(np.squeeze(det[i]))
        else:
            # if multiple faces found, we choose one face
            # which is located center and has larger size
            bounding_box_size = (det[:,2]-det[:,0]) * (det[:,3]-det[:,1])
            img_center = img_size / 2
            offsets = np.vstack([ (det[:,0]+det[:,2])/2 - img_center[0],
                                  (det[:,1]+det[:,3])/2 - img_center[1] ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            # some extra weight on the centering
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            det_arr.append(det[index, :])
            filtered_facial_landmarks = filtered_facial_landmarks[index]
    else:
        det_arr.append(np.squeeze(det))

    for i, det in enumerate(det_arr):
        #-- crop face and recompute the landmark coordinates
        det = np.squeeze(det)
        landmarks = np.squeeze(filtered_facial_landmarks[i])
        # reshape landmarks from (10, ) to (5, 2)
        landmarks = landmarks.reshape(2, 5).T
        
        # compute expanding bounding box
        bb, box_size = get_square_crop_box(det, scalar)
        # compute the relative landmark coordinates using the top-left point
        # of expanding bounding box as ZERO
        landmarks = landmarks - bb[:2]

        # get the valid pixel index of cropped face
        face_left = np.maximum(bb[0], 0)
        face_top = np.maximum(bb[1], 0)
        face_right = np.minimum(bb[2], img_size[0])
        face_bottom = np.minimum(bb[3], img_size[1])
        # cropped square image
        new_img = Image.new('RGB', (box_size, box_size))
        # fullfile the cropped image
        cropped = img.crop([face_left, face_top,
                            face_right, face_bottom])
        w_start_idx = np.maximum(-1*bb[0], 0)
        h_start_idx = np.maximum(-1*bb[1], 0)
        new_img.paste(cropped, (w_start_idx, h_start_idx))
        #new_img = new_img.resize((image_size, image_size), Image.BILINEAR)
            
        #-- face alignment
        # specify size of aligned faces, align and crop with padding
        # due to the bounding box was expanding by a scalar, the `real`
        # face size should be corrected
        scale = box_size * 1.0 / scalar / 112.
        offset = box_size * (scalar - 1.0) / 2
        reference = get_reference_facial_points(default_square=True) * scale \
                    + offset

        warped_face = warp_and_crop_face(np.array(new_img),
                                         landmarks,
                                         reference,
                                         crop_size=(box_size, box_size))
        img_warped = Image.fromarray(warped_face)
        img_warped = img_warped.resize((image_size, image_size), Image.BILINEAR)
        faces.append(img_warped)
    
    return faces


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv.VideoCapture(0)

#-- main
st.title('Face Recognition Demo')

# some constants
SKIP_FRAMES = 10

# initialize variables
total_frames = 0
multitracker = None
ft = FaceTracker(max_disappeared=5)

# init face model
detector = MTCNNDetector(device='cpu')
identity_recognizer = IdentityRecognizer(device='cpu')
gender_recognizer = GenderRecognizer()
age_recognizer = AgeRecognizer()
emotion_recognizer = EmotionRecognizer()

# variables for display
frame_slot = st.empty()
face_slot = st.empty()

cap = get_cap()
while True:
    ret, frame = cap.read()

    # Stop the program if press `q`
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Done processing !!!")
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # crop center square from 1280x960 image
    frame = frame[:, 160:-160, :]
    # BGR to RGB, and switch the left and right side
    im = Image.fromarray(frame[:, ::-1, ::-1])
    im = im.resize((int(im.width/2), int(im.height/2)))


    # check to see if we should run a more computionally expensive
    # face detection method to aid the tracker
    if total_frames % SKIP_FRAMES == 0:
        # face detction
        bounding_boxes, landmarks = detector.infer(im, min_face_size=80)
        
        # create object tracker
        multitracker = cv.MultiTracker_create()
        for bbox in bounding_boxes:
            roi = (int(bbox[0]), int(bbox[1]),
                   int(bbox[2]-bbox[0]), int(bbox[3])-bbox[1])
            multitracker.add(cv.TrackerCSRT_create(), np.array(im), roi)

        # init face info
        faces = []
        features = []
        genders = []
        ages = []
        emotions = []

        # get face info
        if len(bounding_boxes):
            faces = crop_face(im, bounding_boxes, landmarks, scalar=1.2,
                              image_size=227, detect_multiple_faces=True)
            if len(faces):
                features = identity_recognizer.infer(faces)
                genders = gender_recognizer.infer(faces)
                ages = age_recognizer.infer(faces)
                emotions = emotion_recognizer.infer(faces)
 
        # update face tracker
        ft.update(faces, features, genders, ages, emotions)
        faces_img = np.ones((10, 10, 3))
        if len(ft.faces):
            face_ids = list(ft.faces.keys())
            faces_img = [np.array(ft.faces[sel_id].img)
                            for sel_id in face_ids]
            faces_img = np.hstack(tuple(faces_img))
            for face_id in face_ids:
                i = face_ids.index(face_id)
                cv.putText(faces_img,
                           'ID %s: %s'%(face_id, ft.faces[face_id].gender()),
                           (227*i, 22), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 0, 0), 2, cv.LINE_AA)
                cv.putText(faces_img,
                           '%s'%(ft.faces[face_id].age()),
                           (227*i, 50), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 0, 0), 2, cv.LINE_AA)
                cv.putText(faces_img,
                           '%s'%(ft.faces[face_id].emotion()),
                           (227*i, 78), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 0, 0), 2, cv.LINE_AA)

        face_slot.image(faces_img, channels='RGB')

        face_im = show_bboxes(im, bounding_boxes)
        frame_slot.image(np.array(face_im), channels="RGB")
    else:
        if len(multitracker.getObjects()):
            success, boxes = multitracker.update(np.array(im))
            bboxes = []
            for newbox in boxes:
                #print(newbox)
                bboxes.append([int(newbox[0]), int(newbox[1]),
                               int(newbox[0]+newbox[2]),
                               int(newbox[1]+newbox[3])])
            im = show_bboxes(im, bboxes)
        frame_slot.image(np.array(im), channels="RGB")

    total_frames += 1

