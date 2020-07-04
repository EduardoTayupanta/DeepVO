# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
PyCharm - DeepVO
__author__ = 'Eduardo Tayupanta'
__email__ = 'etayupanta@yotec.tech'
"""

# Import Libraries:
import argparse
from dataset import VisualOdometryDataLoader
import matplotlib.pyplot as plt
from TensorFlowImplementation.deepvonet import DeepVONet


def train(model, path, epochs, bsize):
    print('Load Data...')
    dataset = VisualOdometryDataLoader(path, 384, 1280, epochs, bsize, 850)

    print('Summary model...')
    model.summary()

    print('Compile model...')
    model.compile()

    print('Training model...')
    history = model.train(dataset.dataset, len(dataset))
    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])
    plt.show()


def test(model, path):
    print(path)


def main():
    parser = argparse.ArgumentParser(description='TensorFlow DeepVO')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--datapath', default='datapath', type=str, help='path KITII odometry dataset')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
    parser.add_argument('--validation_steps', default=100, type=int, help='test iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    args = parser.parse_args()

    model = DeepVONet(args, 384, 1280)
    if args.mode == 'train':
        train(model, args.datapath, args.train_iter, args.bsize)
    elif args.mode == 'test':
        test(model, args.datapath)

    # --mode train --datapath D:\EduardoTayupanta\Documentos\Librerias\dataset --bsize 2 --lr 0.001 --train_iter 20 --checkpoint_path ./checkpoint


if __name__ == "__main__":
    main()
