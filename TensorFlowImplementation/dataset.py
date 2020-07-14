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
import tensorflow as tf
import matplotlib.pyplot as plt


class VisualOdometryDataLoader:
    def __init__(self, datapath, height, width, batch_size, test=False):
        self.base_path = datapath
        if test:
            self.sequences = ['01']
        else:
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            self.sequences = ['00', '02', '08', '09']

        self.size = 0
        self.sizes = []
        self.poses = self.load_poses()
        self.width = width
        self.height = height

        images_stacked, odometries = self.get_data()

        dataset = tf.data.Dataset.from_tensor_slices((images_stacked, odometries))
        dataset = dataset.shuffle(len(images_stacked))
        dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.dataset = dataset

    def decode_img(self, img):
        image = tf.image.decode_png(img, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.height, self.width])
        return image

    def load_image(self, filename, odometry):
        img1 = tf.io.read_file(filename[0])
        img2 = tf.io.read_file(filename[1])
        img1 = self.decode_img(img1)
        img2 = self.decode_img(img2)
        img = tf.concat([img1, img2], 0)
        return img, odometry

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/', sequence + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
        return all_poses

    def get_image_paths(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        return image_path

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

    def get_data(self):
        images_paths = []
        odometries = []
        for index, sequence in enumerate(self.sequences):
            for i in range(self.sizes[index] - 1):
                images_paths.append([self.get_image_paths(sequence, i), self.get_image_paths(sequence, i + 1)])
                pose1 = self.get6DoFPose(self.poses[index][i])
                pose2 = self.get6DoFPose(self.poses[index][i + 1])
                odom = pose2 - pose1
                odometries.append(odom)
        return np.array(images_paths), np.array(odometries)

    def __len__(self):
        return self.size - len(self.sequences)


def main():
    path = "D:\EduardoTayupanta\Documents\Librerias\dataset"
    dataset = VisualOdometryDataLoader(path, 384, 1280, 32)
    for element in dataset.dataset.as_numpy_iterator():
        for index in range(len(element[0])):
            img = element[0][index]
            print(element[1][index])
            plt.title("TensorFlow Logo with shape {}".format(img.shape))
            _ = plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    main()
