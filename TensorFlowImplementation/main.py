# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
from dataset import VisualOdometryDataLoader
from deepvonet import DeepVONet, FlowNet
import matplotlib.pyplot as plt
import tensorflow as tf


# Custom loss function.
def custom_loss(y_pred, y_true, k, criterion):
    mse_position = criterion(y_true[:, :3], y_pred[:, :3])
    mse_orientation = criterion(y_true[:, 3:], y_pred[:, 3:])
    return mse_position + k * mse_orientation


def run_optimization(model, x, y, k, criterion, optimizer):
    with tf.GradientTape() as g:
        # Forward pass.
        pred = model(x, is_training=True)
        # Compute loss.
        loss = custom_loss(pred, y, k, criterion)

    # Variables to update, i.e. trainable variables.
    trainable_variables = model.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def train_model(dataset, flownet, deepvonet, config, criterion, optimizer, epoch):
    loss = 0.0
    for step, (batch_x, batch_y) in enumerate(dataset.dataset):
        with tf.device('/gpu:0'):
            flow = flownet(batch_x)
        with tf.device('/cpu:0'):
            run_optimization(deepvonet, flow, batch_y, config['k'], criterion, optimizer)
            pred = deepvonet(flow)

        loss = custom_loss(pred, batch_y, config['k'], criterion).numpy()
        print('Epoch {}, \t Step: {}, \t Loss: {}'.format(epoch, step, loss))
    return loss


def train(flownet, deepvonet, config):
    print('Load Data...')
    dataset = VisualOdometryDataLoader(config['datapath'], 384, 1280, config['bsize'])

    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=config['lr'], momentum=config['momentum'], nesterov=True)

    print('Training model...')
    total_loss = []
    for epoch in range(1, config['train_iter']):
        loss = train_model(dataset, flownet, deepvonet, config, criterion, optimizer, epoch)
        flownet.save_weights(config['checkpoint_path'] + '/deepvo-{:04d}.ckpt'.format(epoch))
        total_loss.append(loss)

    print('Plot loss...')
    fig, ax = plt.subplots()
    ax.plot(range(len(total_loss)), total_loss)

    ax.set(xlabel='Epoch Number', ylabel='Loss Magnitude', title='Loss per epoch')
    ax.grid()

    fig.savefig("loss.png")
    plt.show()


def test(model, path):
    print(path)


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    config = {
        'mode': 'train',
        'datapath': 'D:\EduardoTayupanta\Documents\Librerias\dataset',
        'bsize': 8,
        'lr': 0.001,
        'momentum': 0.99,
        'train_iter': 20,
        'checkpoint_path': './checkpoints',
        'k': 100,
    }

    deepvonet = DeepVONet()
    flownet = FlowNet()

    if config['mode'] == 'train':
        train(flownet, deepvonet, config)
    elif config['mode'] == 'test':
        test(deepvonet, config)


if __name__ == "__main__":
    main()
