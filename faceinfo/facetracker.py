# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)',
            '(38, 43)', '(48, 53)', '(60, 100)']
EMOTION_TABLE = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
                 4: 'Anger', 5: 'Disgust', 6: 'Fear', 7: 'Contemp'}

def corrcoef2(A, B):
    """Row-wise Correlation Coefficient calculation for two 2D arrays.
    Input 2D array A (m by d), array B (n by d), the function would return
    a correlation matrix (m by n).
    """
    # Row-wise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coef
    return np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:, None], ssB[None]))


class TrackableFace():
    """Trackable Face Object."""
    def __init__(self, face_img, face_feat):
        # store the face's unique face feature,
        # then initialize a list of emotion status
        self.img = face_img
        self.face_feat = face_feat
        self.age_probs = None
        self.gender_probs = None
        self.emotions = []

    def update_img(self, face_img):
        self.img = face_img

    def update_age(self, age_probs):
        if isinstance(self.age_probs, np.ndarray):
            self.age_probs = 0.5 * self.age_probs + 0.5 * age_probs
        else:
            self.age_probs = age_probs

    def update_gender(self, gender_probs):
        if isinstance(self.gender_probs, np.ndarray):
            self.gender_probs = 0.5 * self.gender_probs + 0.5 * gender_probs
        else:
            self.gender_probs = gender_probs

    def update_emotion(self, emotion_idx):
        self.emotions.append(EMOTION_TABLE[emotion_idx])

    def age(self):
        return AGE_LIST[self.age_probs.argmax()]

    def gender(self):
        return GENDER_LIST[self.gender_probs.argmax()]

    def emotion(self):
        return self.emotions[-1]


class FaceTracker():
    def __init__(self, max_disappeared=20):
        # initialize the next unique face ID along with two ordered
        # dictionaries used to keep track of mapping a given face
        # ID to its face-index and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.next_face_id = 0
        self.faces = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.max_disappeared = max_disappeared

    def register(self, face_img, face_feature):
        # when registering a face we use the next available face
        # ID to store the face info
        self.faces[self.next_face_id] = TrackableFace(face_img, face_feature)
        self.disappeared[self.next_face_id] = 0
        self.next_face_id += 1

        # return current added face_id
        return self.next_face_id-1

    def deregister(self, face_id):
        # to deregister a face ID we delete the face ID from
        # both of our respective dictionaries
        del self.faces[face_id]
        del self.disappeared[face_id]

    def update(self, face_imgs, face_features, genders=[],
               ages=[], emotions=[]):
        # check to see if the array of input face features is empty
        if len(face_features)==0:
            # loop over any existing tracked faces and mark them as disappeared
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1

	        # if we have reached a maximum number of consecutive frames
                # where a given face has been marked as missing, deregister it
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)

            # return early as there are no tracking info to update
            return self.faces

        # if we are currently not tracking any faces take the input
        # infos and register each of them
        if len(self.faces) == 0:
            for i in range(len(face_features)):
                current_id = self.register(face_imgs[i], face_features[i])
                if len(genders):
                    self.faces[current_id].update_gender(genders[i])
                if len(ages):
                    self.faces[current_id].update_age(ages[i])
                if len(emotions):
                    self.faces[current_id].update_emotion(emotions[i])

        # otherwise, we are currently tracking faces so we need to
        # try to match the input face features to existing faces'
        else:
            # grab the set of face IDs and corresponding face features
            face_ids = list(self.faces.keys())
            tracked_features = np.array([self.faces[i].face_feat 
                                            for i in face_ids])

            # compute the corrcoef between each pair of tracked face features
            # and input face features, respectively -- our goal will be to
            # match an input face to a tracked face
            R = corrcoef2(tracked_features, face_features)

            # in order to perform this matching we must (1) find the largest
            # value in each row and then (2) sort the row indexes based on
            # their maxmium values so that the row with the largest value is
            # at the *front* of the index list
            rows = R.max(axis=1).argsort()[::-1]

            # next, we find the column index of the largest value in each row
            cols = R.argmax(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister a face we need to keep track of which
            # of the row and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it val
                if row in usedRows or col in usedCols:
                    continue

                # check the similarity between tracked face and the input
                # if they are same
                if R[row, col]>0.5:
                    # grab the face ID for the current row, update its new
                    # info, and reset the disappeared counter
                    sel_id = face_ids[row]
                    self.faces[sel_id].face_feat = face_features[col]
                    self.faces[sel_id].update_img(face_imgs[col])
                    if len(genders):
                        self.faces[sel_id].update_gender(genders[col])
                    if len(ages):
                        self.faces[sel_id].update_age(ages[col])
                    if len(emotions):
                        self.faces[sel_id].update_emotion(emotions[col])
                    self.disappeared[sel_id] = 0

                    # indicate that we have examined each of the row and
                    # column indexes, respectively
                    usedRows.add(row)
                    usedCols.add(col)
 
            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, R.shape[0])).difference(usedRows)
            unusedCols = set(range(0, R.shape[1])).difference(usedCols)

            # for unused row index (we does not find the face in new frame),
            # we need to check and see if some of these faces have
            # potentially disappeared
            for row in unusedRows:
                # grab the face ID for the corresponding row
                # index and increment the disappeared counter
                sel_id = face_ids[row]
                self.disappeared[sel_id] += 1

                # check to see if the number of consecutive
                # frames the face has been marked "disappeared"
                # for warrants deregistering the face
                if self.disappeared[sel_id] > self.max_disappeared:
                    self.deregister(sel_id)

            # for unused column index (maybe new face appears),
            # we need to register each new input face as a trackable object
            for col in unusedCols:
                current_id = self.register(face_imgs[col], face_features[col])
                if len(genders):
                    self.faces[current_id].update_gender(genders[col])
                if len(ages):
                    self.faces[current_id].update_age(ages[col])
                if len(emotions):
                    self.faces[current_id].update_emotion(emotions[col])

        # return the set of trackable faces
        return self.faces

