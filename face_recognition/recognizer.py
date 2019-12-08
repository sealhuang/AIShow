# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
from PIL import Image
import cv2 as cv
import torch
import torchvision.transforms as transforms
import onnxruntime as rt

from faceinsight.models.shufflenet_v2 import ShuffleNetV2


def softmax(scores):
    """Compute softmax values for each row of scores."""
    s = scores.T
    e_s = np.exp(s - np.max(s))
    softmax_v = e_s / e_s.sum(axis=0)
    return softmax_v.T


class IdentityRecognizer():
    """Face Identity Recognizer based on ShuffleNet."""
    
    def __init__(self, device='cpu'):
        """Model initialization."""
        self.device = device

        # image preprocessing config
        RGB_MEAN = [0.5, 0.5, 0.5]
        RGB_STD = [0.5, 0.5, 0.5]
        self.img_transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=RGB_MEAN,
                                                         std=RGB_STD),
                            ])

        # load model
        model_file = 'data/Backbone_shufflenet_v2_x1_0_Epoch_40_checkpoint.pth'
        self.model = ShuffleNetV2(n_class=512, input_size=224, width_mult=1.0)
        self.model.load_state_dict(torch.load(model_file,
                                map_location=lambda storage, loc: storage))
        if device=='gpu':
            self.model = self.model.cuda()
        self.model.eval()

    def infer(self, imgs):
        """Input:
            `imgs`: list of PIL.Image.
        
        Return:
            `face_features`: numpy.array of shape [n_faces, embedding_size]
        """
        # image transformation
        inputs = []
        for sample in imgs:
            inputs.append(self.img_transform(sample))
        inputs = torch.stack(inputs)

        # compute output
        inputs = inputs.to(self.device)
        features = self.model(inputs)
        #print(features.cpu().data.numpy().shape)

        return features.cpu().data.numpy()


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

class GenderRecognizer():
    """Gender Recognizer."""

    def __init__(self):
        """Model initialization."""
        self.model = cv.dnn.readNetFromCaffe('data/deploy_gender.prototxt',
                                             'data/gender_net.caffemodel')

    def infer(self, imgs):
        """Input:
            `imgs`: list of PIL.Image, image size should be (227, 227, 3)
        
        Return:
            list of gender probabilities, each element is a vector of
            shape (2, ), i.e. [`male probability`, 'female probability'],
            for one face.
        """
        genders = []
        for i in range(len(imgs)):
            f = np.array(imgs[i])
            blob = cv.dnn.blobFromImage(f[:, :, ::-1], 1, (227, 227),
                                        MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            self.model.setInput(blob)
            gender_preds = self.model.forward()
            genders.append(gender_preds[0])

        return genders


class AgeRecognizer():
    """Age Recognizer."""

    def __init__(self):
        """Model initialization.

        Label of the model:
            AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)',
                        '(38, 43)', '(48, 53)', '(60, 100)']
        """
        self.model = cv.dnn.readNetFromCaffe('data/deploy_age.prototxt',
                                             'data/age_net.caffemodel')

    def infer(self, imgs):
        """Input:
            `imgs`: list of PIL.Image, image size should be (227, 227, 3)
        
        Return:
            list of age probabilities.
        """
        ages = []
        for i in range(len(imgs)):
            f = np.array(imgs[i])
            blob = cv.dnn.blobFromImage(f[:, :, ::-1], 1, (227, 227),
                                        MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            self.model.setInput(blob)
            age_preds = self.model.forward()
            ages.append(age_preds[0])

        return ages


class EmotionRecognizer():
    """Emotion Recognizer."""

    def __init__(self):
        """Model initialization.
        The model outputs a (1x8) array of scores corresponding to the 8
        emotion classes, where the labels map as follows:
        emotion_table = {'neutral': 0, 'happiness': 1, 'surprise': 2,
                         'sadness': 3, 'anger': 4, 'disgust': 5,
                         'fear': 6, 'contempt': 7}
        """
        self.sess = rt.InferenceSession('data/emotion_model_ferplus.onnx')
        self.input_name = self.sess.get_inputs()[0].name

    def infer(self, imgs):
        """Input:
            `imgs`: list of PIL.Image, image size should be (x, x, 3)
        
        Return:
            list of age probabilities.
        """
        scores = []
        for i in range(len(imgs)):
            img = imgs[i].convert('L')
            img = img.resize((64, 64), Image.ANTIALIAS)
            img_data = np.array(img)
            img_data = np.resize(img_data, (1, 1, 64, 64))
            score = self.sess.run(None,
                            {self.input_name: img_data.astype(np.float32)})[0]
            scores.append(np.squeeze(score))

        scores = np.array(scores)
        probs = softmax(scores)
        class_idxes = np.argsort(probs, axis=1)[:, -1]

        return list(class_idxes)

