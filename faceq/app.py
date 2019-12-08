# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtWidgets, QtGui

from faceinsight.detection import MTCNNDetector, show_bboxes
from faceinsight.detection.align_trans import get_reference_facial_points
from faceinsight.detection.align_trans import warp_and_crop_face


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(Image.Image)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        ret, frame = self.camera.read()
        if ret:
            # crop center square from 1280x960 image
            frame = frame[:, 160:-160, :]
            # BGR to RGB, and switch the left and right side
            im = Image.fromarray(frame[:, ::-1, ::-1])
            self.image_data.emit(im.resize((int(im.width/2), int(im.height/2))))


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, detector_device='cpu', parent=None):
        super().__init__(parent)
        self.detector = MTCNNDetector(device=detector_device)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = 90

    def detect_faces(self, img: Image.Image):
        bounding_boxes, landmarks = self.detector.infer(img,
                                        min_face_size=self._min_size)
        
        faces = self._crop_face(img, bounding_boxes, landmarks, scalar=1.5,
                                image_size=227, detect_multiple_faces=True)

        return bounding_boxes, faces

    def _get_square_crop_box(self, crop_box, box_scalar=1.0):
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

    def _crop_face(self, img, bounding_boxes, facial_landmarks, scalar,
                   image_size, detect_multiple_faces=True):
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
            bb, box_size = self._get_square_crop_box(det, scalar)
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
            reference = get_reference_facial_points(default_square=True)*scale \
                        + offset

            warped_face = warp_and_crop_face(np.array(new_img),
                                             landmarks,
                                             reference,
                                             crop_size=(box_size, box_size))
            img_warped = Image.fromarray(warped_face)
            img_warped = img_warped.resize((image_size, image_size),
                                           Image.BILINEAR)
            faces.append(img_warped)
    
        return faces

        def image_data_slot(self, image_data):
                faces = self.detect_faces(image_data)
                for (x, y, w, h) in faces:
                        cv2.rectangle(image_data, (x, y), (x+w, y+h), self._red, self._width)

                self.image = self.get_qimage(image_data)
                if self.image.size() != self.size():
                        self.setFixedSize(self.image.size())

                self.update()

        def get_qimage(self, image: np.ndarray):
                height, width, colors = image.shape
                bytesPerLine = 3 * width
                QImage = QtGui.QImage

                image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

                image = image.rgbSwapped()
                return image

        def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                painter.drawImage(0, 0, self.image)
                self.image = QtGui.QImage()

    def image_data_slot(self, img):
        bounding_boxes, faces = self.detect_faces(img)
        bboxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    for b in bounding_boxes]

        image_data = np.array(img)[:, :, ::-1]
        image_data = image_data.astype(np.uint8)

        for b in bboxes:
            cv2.rectangle(image_data,
                          (b[0], b[1]),
                          (b[2], b[3]),
                          self._red,
                          self._width)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data, width, height, bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, detector_device='cpu', parent=None):
        super().__init__(parent)
        self.face_detection_widget = FaceDetectionWidget(detector_device)

        # TODO: set video port
        self.record_video = RecordVideo()
        self.run_button = QtWidgets.QPushButton('Start')

        # Connect the image data signal and slot together
        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)
        # connect the run button to the start recording slot
        self.run_button.clicked.connect(self.record_video.start_recording)

        # Create and set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.face_detection_widget)
        layout.addWidget(self.run_button)

        self.setLayout(layout)


def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(detector_device='cpu')
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

