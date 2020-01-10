# vi: set ft=python sts=4 ts=4 sw=4 et:

import sys
import os
import time

import numpy as np
import cv2
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from faceinsight.detection import MTCNNDetector, show_bboxes, show_grids

from utils import Config
from model import Model
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene

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

def face_highlight(img, bounding_boxes, scalar):
    """Give a highlight to the largest face in the image.

    Return:
        A face-highlighted version of input image.
    """
    img = img.astype('int32')
    # filter real faces based on detection confidence
    confidence_thresh = 0.85
    filtered_idx = bounding_boxes[:, 4]>=confidence_thresh
    filtered_bboxes = bounding_boxes[filtered_idx]

    # if no faces found, return a darker image
    if not len(filtered_bboxes):
        return np.clip(img-50, 0, 255).astype('uint8')

    nrof_faces = len(filtered_bboxes)

    # detect multiple faces or not
    det = filtered_bboxes[:, 0:4]
    det_arr = []
    img_size = np.asarray(img.shape)
    if nrof_faces>1:
        # if multiple faces found, we choose one face
        # which is located center and has larger size
        bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([(det[:,0]+det[:,2])/2 - img_center[0],
                             (det[:,1]+det[:,3])/2 - img_center[1]])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        # some extra weight on the centering
        index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
        det_arr.append(det[index, :])
    else:
        det_arr.append(np.squeeze(det))

    det = np.squeeze(det_arr)
    # compute expanding bounding box
    bb, box_size = get_square_crop_box(det, scalar)
    # get the valid pixel index of cropped face
    face_left = np.maximum(bb[0], 0)
    face_top = np.maximum(bb[1], 0)
    face_right = np.minimum(bb[2], img_size[0])
    face_bottom = np.minimum(bb[3], img_size[1])

    # highlight the face with circle
    xx, yy = np.mgrid[:img_size[0], :img_size[1]]
    center_x = int((bb[3] + bb[1])/2)
    center_y = int((bb[2] + bb[0])/2)
    circle_r2 = int(0.25 * box_size**2)
    circle = (xx - center_x) ** 2 + (yy - center_y) ** 2
    highlight_mat = circle > circle_r2
    highlight_mat = np.repeat(np.expand_dims(highlight_mat, 2), 3, axis=2)
    # highlight the face with square
    #highlight_mat = np.ones_like(img)
    #highlight_mat[face_top:face_bottom, face_left:face_right, :] = 0
    
    return np.clip(img-50*highlight_mat, 0, 255).astype('uint8')

def crop_face(img, bounding_boxes, scalar):
    """Crop face from input image.

    Return:
        cropped face image.
    """
    # filter real faces based on detection confidence
    confidence_thresh = 0.85
    filtered_idx = bounding_boxes[:, 4]>=confidence_thresh
    filtered_bboxes = bounding_boxes[filtered_idx]

    # if no faces found, return a darker image
    if not len(filtered_bboxes):
        return []

    nrof_faces = len(filtered_bboxes)

    det = filtered_bboxes[:, 0:4]
    det_arr = []
    img_size = np.asarray(img.shape)
    if nrof_faces>1:
        # if multiple faces found, we choose one face
        # which is located center and has larger size
        bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([(det[:,0]+det[:,2])/2 - img_center[0],
                             (det[:,1]+det[:,3])/2 - img_center[1]])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        # some extra weight on the centering
        index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
        det_arr.append(det[index, :])
    else:
        det_arr.append(np.squeeze(det))

    det = np.squeeze(det_arr)
    # compute expanding bounding box
    bb, box_size = get_square_crop_box(det, scalar)
    # get the valid pixel index of cropped face
    face_left = np.maximum(bb[0], 0)
    face_top = np.maximum(bb[1], 0)
    face_right = np.minimum(bb[2], img_size[0])
    face_bottom = np.minimum(bb[3], img_size[1])
    # cropped square image
    new_img = Image.new('RGB', (box_size, box_size))
    # fullfile the cropped image
    img = Image.fromarray(img)
    cropped = img.crop([face_left, face_top,
                        face_right, face_bottom])
    w_start_idx = np.maximum(-1*bb[0], 0)
    h_start_idx = np.maximum(-1*bb[1], 0)
    new_img.paste(cropped, (w_start_idx, h_start_idx))
    new_img = new_img.resize((512, 512), Image.BILINEAR)
    
    return [np.array(new_img)]


class RecordVideo(QtCore.QObject):
    frame_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(3, 1280)
        self.camera.set(4, 720)
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def end_recording(self):
        self.timer.stop()
        self.camera.release()

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        ret, frame = self.camera.read()
        if ret:
            # crop center square from 720x1280 image
            frame = frame[:, 280:-280, :]
            # switch the left and right side
            frame = frame[:, ::-1, :]

            self.frame_data.emit(frame)


class Ex(QtWidgets.QWidget, Ui_Form):
    def __init__(self, model, config):
        super().__init__()

        #self.get_head_outline()
        self._step_counter = 1

        # start camera
        self.record_video = RecordVideo()
        # connect the frame data signal and slot together
        self.record_video.frame_data.connect(self.camera_data_slot)

        # start face detector
        self.detector = MTCNNDetector(device='cpu')

        self.setupUi(self)
        self.show()
        self.model = model
        self.config = config
        self.model.load_demo_graph(config)

        self.output_img = None

        self.mat_img = None

        self.ld_mask = None
        self.ld_sk = None

        self._frame_data = None

        self.modes = [0, 0, 0]
        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
        )
        self.graphicsView.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.graphicsView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )

        self.result_scene = QtWidgets.QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
        )
        self.graphicsView_2.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.graphicsView_2.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )

        self.dlg = QtWidgets.QColorDialog(self.graphicsView)
        self.color = None

    def get_head_outline(self):
        """load head outline."""
        head_outline = cv2.imread('./ui/head_outline.jpg')
        head_outline = head_outline[25:-125, 75:-75, :]
        head_outline = cv2.resize(head_outline, (512, 512),
                                  interpolation=cv2.INTER_CUBIC)
        head_outline[head_outline<75] = 0.5
        head_outline[head_outline>1] = 1
        self._head_outline = head_outline

    def mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def camera_data_slot(self, frame_data):
        self._step_counter += 1

        self._frame_data = frame_data

        # resize to 512x512
        frame_data = cv2.resize(frame_data, (512, 512),
                                interpolation=cv2.INTER_CUBIC)

        # face detection
        # BGR to RGB first
        im = Image.fromarray(frame_data[:, :, ::-1])
        bboxes, landmarks = self.detector.infer(im, min_face_size=200)
        if len(bboxes):
            _img = show_grids(im, [], landmarks, step=self._step_counter%3)
            frame_data  = np.array(_img)[:, :, ::-1]
            frame_data = face_highlight(frame_data, bboxes, 1.2)
        else:
            frame_data = frame_data.astype('int32')
            frame_data = np.clip(frame_data-50, 0, 255).astype('uint8')

        # convert data frame into QImage
        h, w, c = frame_data.shape
        bytes_per_line = 3 * w
        _frame_image = QtGui.QImage(frame_data.data, w, h, bytes_per_line,
                                    QtGui.QImage.Format_RGB888)
        _frame_image = _frame_image.rgbSwapped()

        # draw frame
        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()
        self.scene.addPixmap(QtGui.QPixmap.fromImage(_frame_image))

    def capture(self):
        self.record_video.timer.stop()
        if self._frame_data is None:
            return
       
        # face detection
        # BGR to RGB first
        im = Image.fromarray(self._frame_data[:, :, ::-1])
        bboxes, _ = self.detector.infer(im, min_face_size=200)
        if len(bboxes):
            faces = crop_face(self._frame_data, bboxes, 1.4)
            #print(lmarks)
        else:
            return

        if len(faces)==0:
            return

        # draw landmarks on display image
        _face_img = Image.fromarray(faces[0][:, :, ::-1])
        _, landmarks = self.detector.infer(_face_img, min_face_size=100)
        #_face_img = show_bboxes(_face_img, [], landmarks)
        _face_img = show_grids(_face_img, [], landmarks)
        _face_img = np.array(_face_img)[:, :, ::-1]

        # convert data frame into QImage
        self._frame_data = faces[0]
        h, w, c = self._frame_data.shape
        bytes_per_line = 3 * w
        #_frame_image = QtGui.QImage(self._frame_data.data, w, h,
        #                            bytes_per_line,
        #                            QtGui.QImage.Format_RGB888)
        _frame_image = QtGui.QImage(_face_img.copy(), w, h,
                                    bytes_per_line,
                                    QtGui.QImage.Format_RGB888)
        _frame_image = _frame_image.rgbSwapped()
        image = QtGui.QPixmap.fromImage(_frame_image)
        mat_img = self._frame_data

        self.image = image.scaled(self.graphicsView.size(),
                                  QtCore.Qt.IgnoreAspectRatio)
        mat_img = mat_img/127.5 - 1
        self.mat_img = np.expand_dims(mat_img, axis=0)
        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()
        self.scene.addPixmap(self.image)
        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(self.image)

    def open(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File",
            QtCore.QDir.currentPath()
        )
        if fileName:
            image = QtGui.QPixmap(fileName)
            mat_img = cv2.imread(fileName)
            if image.isNull():
                QtWidgets.QMessageBox.information(
                    self,
                    "Image Viewer",
                    "Cannot load %s." % fileName
                )
                return

            # redbrush = QBrush(Qt.red)
            # blackpen = QPen(Qt.black)
            # blackpen.setWidth(5)
            self.image = image.scaled(self.graphicsView.size(),
                                      QtCore.Qt.IgnoreAspectRatio)
            mat_img = cv2.resize(mat_img,
                                 (512, 512),
                                 interpolation=cv2.INTER_CUBIC)
            mat_img = mat_img/127.5 - 1
            self.mat_img = np.expand_dims(mat_img, axis=0)
            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()
            self.scene.addPixmap(self.image)
            if len(self.result_scene.items()) > 0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(self.image)

    def mask_mode(self):
        self.mode_select(0)

    def sketch_mode(self):
        self.mode_select(1)

    def stroke_mode(self):
        if not self.color:
            self.color_change_mode()
        self.scene.get_stk_color(self.color)
        self.mode_select(2)

    def color_change_mode(self):
        self.dlg.exec_()
        self.color = self.dlg.currentColor().name()
        self.pushButton_4.setStyleSheet("background-color: %s;" % self.color)
        self.scene.get_stk_color(self.color)

    def complete(self):
        sketch = self.make_sketch(self.scene.sketch_points)
        stroke = self.make_stroke(self.scene.stroke_points)
        mask = self.make_mask(self.scene.mask_points)
        if not type(self.ld_mask)==type(None):
            ld_mask = np.expand_dims(self.ld_mask[:,:,0:1], axis=0)
            ld_mask[ld_mask>0] = 1
            ld_mask[ld_mask<1] = 0
            mask = mask + ld_mask
            mask[mask>0] = 1
            mask[mask<1] = 0
            mask = np.asarray(mask, dtype=np.uint8)
            print(mask.shape)

        if not type(self.ld_sk)==type(None):
            sketch = sketch + self.ld_sk
            sketch[sketch>0] = 1 

        noise = self.make_noise()

        sketch = sketch * mask
        stroke = stroke * mask
        noise = noise * mask

        batch = np.concatenate([self.mat_img, sketch, stroke, mask, noise],
                               axis=3)
        start_t = time.time()
        result = self.model.demo(self.config, batch)
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        result = (result + 1) * 127.5
        result = np.asarray(result[0, :, :, :], dtype=np.uint8)
        self.output_img = result
        result = np.concatenate([
                result[:, :, 2:3],
                result[:, :, 1:2],
                result[:, :, :1]
            ],
            axis=2)
        qim = QtGui.QImage(result.data, result.shape[1], result.shape[0],
                           result.strides[0], QtGui.QImage.Format_RGB888)
        self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QtGui.QPixmap.fromImage(qim))

    def make_noise(self):
        noise = np.zeros([512, 512, 1], dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise/255, dtype=np.uint8)
        noise = np.expand_dims(noise, axis=0)
        return noise

    def make_mask(self, pts):
        if len(pts)>0:
            mask = np.zeros((512, 512, 3))
            for pt in pts:
                cv2.line(mask, pt['prev'], pt['curr'], (255,255,255), 12)
            mask = np.asarray(mask[:, :, 0]/255, dtype=np.uint8)
            mask = np.expand_dims(mask, axis=2)
            mask = np.expand_dims(mask, axis=0)
        else:
            mask = np.zeros((512, 512, 3))
            mask = np.asarray(mask[:, :, 0]/255, dtype=np.uint8)
            mask = np.expand_dims(mask, axis=2)
            mask = np.expand_dims(mask, axis=0)
        return mask

    def make_sketch(self, pts):
        if len(pts)>0:
            sketch = np.zeros((512,512,3))
            # sketch = 255*sketch
            for pt in pts:
                cv2.line(sketch,pt['prev'],pt['curr'],(255,255,255),1)
            sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        else:
            sketch = np.zeros((512,512,3))
            # sketch = 255*sketch
            sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        return sketch

    def make_stroke(self, pts):
        if len(pts)>0:
            stroke = np.zeros((512,512,3))
            for pt in pts:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i+2], 16) for i in (0, 2 ,4))
                color = (color[2],color[1],color[0])
                cv2.line(stroke,pt['prev'],pt['curr'],color,4)
            stroke = stroke/127.5 - 1
            stroke = np.expand_dims(stroke,axis=0)
        else:
            stroke = np.zeros((512,512,3))
            stroke = stroke/127.5 - 1
            stroke = np.expand_dims(stroke,axis=0)
        return stroke

    def arrange(self):
        image = np.asarray((self.mat_img[0]+1)*127.5,dtype=np.uint8)
        if len(self.scene.mask_points)>0:
            for pt in self.scene.mask_points:
                cv2.line(image,pt['prev'],pt['curr'],(255,255,255),12)
        if len(self.scene.stroke_points)>0:
            for pt in self.scene.stroke_points:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i+2], 16) for i in (0, 2 ,4))
                color = (color[2],color[1],color[0])
                cv2.line(image,pt['prev'],pt['curr'],color,4)
        if len(self.scene.sketch_points)>0:
            for pt in self.scene.sketch_points:
                cv2.line(image,pt['prev'],pt['curr'],(0,0,0),1)        
        cv2.imwrite('tmp.jpg',image)
        image = QtGui.QPixmap('tmp.jpg')
        self.scene.history.append(3)
        self.scene.addPixmap(image)

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save File",
                QtCore.QDir.currentPath()
            )
            cv2.imwrite(fileName+'.jpg', self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)


if __name__ == '__main__':
    config = Config('demo.yaml')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_NUM)
    model = Model(config)

    app = QtWidgets.QApplication(sys.argv)
    ex = Ex(model, config)
    sys.exit(app.exec_())

