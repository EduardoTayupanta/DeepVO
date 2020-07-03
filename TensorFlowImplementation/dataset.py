# -*- coding: utf-8 -*-
"""
Created by etayupanta at 7/1/2020 - 16:11
PyCharm - DeepVO
__author__ = 'Eduardo Tayupanta'
__email__ = 'etayupanta@yotec.tech'
"""

# Import Libraries:
import math
import numpy as np
import os
from PIL import Image


def default_image_loader(path):
    return Image.open(path).convert('RGB')  # .transpose(0, 2, 1)


def default_image_preprocess(img, width, height):
    return img.resize((width, height), Image.ANTIALIAS)


class VisualOdometryDataLoader:
    def __init__(self, datapath, height, width, transform=default_image_preprocess, test=False,
                 loader=default_image_loader):
        self.base_path = datapath
        if test:
            self.sequences = ['01']
        else:
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            self.sequences = ['00']

        self.size = 0
        self.sizes = []
        self.poses = self.load_poses()

        self.transform = transform
        self.loader = loader
        self.width = width
        self.height = height

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/', sequence + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
        return all_poses

    def get_image(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        image = self.loader(image_path)
        return image

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z], dtype=np.float32)

    def get6DoFPose(self, p):
        pos = np.array([p[3], p[7], p[11]])
        R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
        angles = self.rotationMatrixToEulerAngles(R)
        return np.concatenate((pos, angles))

    def __len__(self):
        return self.size - len(self.sequences)

    def __getitem__(self, index):
        sequence = 0
        img1 = self.get_image(self.sequences[sequence], index)
        img2 = self.get_image(self.sequences[sequence], index + 1)
        pose1 = self.get6DoFPose(self.poses[sequence][index])
        pose2 = self.get6DoFPose(self.poses[sequence][index + 1])
        odom = pose2 - pose1
        if self.transform is not None:
            img1 = self.transform(img1, self.width, self.height)
            img2 = self.transform(img2, self.width, self.height)
        return np.concatenate([img1, img2]), odom


def load_train_set(dirname, verbose=True):
    dataset = VisualOdometryDataLoader(dirname, 384, 1280)
    X_train = []
    y_train = []
    # for i in range(len(dataset)):
    for i in range(10):
        image, odom = dataset[i]
        X_train.append(image)
        y_train.append(odom)
        if i % 10 == 1:
            print('.', end='')
        else:
            print('')
    return np.array(X_train), np.array(y_train)


def main():
    X, y = load_train_set("D:\EduardoTayupanta\Documentos\Librerias\dataset")
    X = X.astype('float32') / 255.0
    print(y)
    # perm = np.random.permutation(len(X))
    # X, y = X[perm], y[perm]
    # dataset = VisualOdometryDataLoader("D:\EduardoTayupanta\Documentos\Librerias\dataset", 384, 1280)
    # X, y = dataset[0]
    # print(X)


if __name__ == "__main__":
    main()
