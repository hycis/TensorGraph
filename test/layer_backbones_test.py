import tensorgraphx as tg
from tensorgraphx.layers.backbones import *
from tensorgraphx.layers import Softmax, Flatten, Linear, MaxPooling
import tensorflow as tf
import os
from tensorgraphx.trainobject import train as mytrain
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


X_train = np.random.rand(10, 32, 32, 1)
y_train = np.random.rand(10, 1)
_, h, w, c = X_train.shape
_, nclass = y_train.shape
X_ph = tf.placeholder('float32', [None, h, w, c])
y_ph = tf.placeholder('float32', [None, nclass])


def train(seq):
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)
    train_cost_sb = tg.cost.entropy(y_ph, y_train_sb)
    optimizer = tf.train.AdamOptimizer(0.0001)
    test_accu_sb = tg.cost.accuracy(y_ph, y_test_sb)
    with tf.Session() as sess:
        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=1,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=1, randomize_split=False)


def test_VGG16():
    seq = tg.Sequential()
    vgg = VGG16(input_channels=c, input_shape=(h, w))
    print('output channels:', vgg.output_channels)
    print('output shape:', vgg.output_shape)
    out_dim = np.prod(vgg.output_shape) * vgg.output_channels
    seq.add(vgg)
    seq.add(Flatten())
    seq.add(Linear(int(out_dim), nclass))
    seq.add(Softmax())
    train(seq)


def test_VGG19():
    seq = tg.Sequential()
    vgg = VGG19(input_channels=c, input_shape=(h, w))
    print('output channels:', vgg.output_channels)
    print('output shape:', vgg.output_shape)
    out_dim = np.prod(vgg.output_shape) * vgg.output_channels
    seq.add(vgg)
    seq.add(Flatten())
    seq.add(Linear(int(out_dim), nclass))
    seq.add(Softmax())
    train(seq)


def test_ResNetSmall():
    seq = tg.Sequential()
    model = ResNetSmall(input_channels=c, input_shape=(h, w), config=[1,1])
    model = ResNetBase(input_channels=c, input_shape=(h, w), config=[1,1,1,1])
    print('output channels:', model.output_channels)
    print('output shape:', model.output_shape)
    seq.add(model)
    seq.add(MaxPooling(poolsize=tuple(model.output_shape), stride=(1,1), padding='VALID'))
    outshape = valid_nd(model.output_shape, kernel_size=model.output_shape, stride=(1,1))
    print(outshape)
    out_dim = model.output_channels
    seq.add(Flatten())
    seq.add(Linear(int(out_dim), nclass))
    seq.add(Softmax())
    train(seq)


def test_ResNetBase():
    seq = tg.Sequential()
    model = ResNetBase(input_channels=c, input_shape=(h, w), config=[1,1,1,1])
    print('output channels:', model.output_channels)
    print('output shape:', model.output_shape)
    seq.add(model)
    seq.add(MaxPooling(poolsize=tuple(model.output_shape), stride=(1,1), padding='VALID'))
    outshape = valid_nd(model.output_shape, kernel_size=model.output_shape, stride=(1,1))
    print(outshape)
    out_dim = model.output_channels
    seq.add(Flatten())
    seq.add(Linear(int(out_dim), nclass))
    seq.add(Softmax())
    train(seq)


def test_DenseNet():
    seq = tg.Sequential()
    model = DenseNet(input_channels=c, input_shape=(h, w), ndense=1, growth_rate=1, nlayer1blk=1)
    print('output channels:', model.output_channels)
    print('output shape:', model.output_shape)
    seq.add(model)
    seq.add(MaxPooling(poolsize=tuple(model.output_shape), stride=(1,1), padding='VALID'))
    seq.add(Flatten())
    seq.add(Linear(model.output_channels, nclass))
    seq.add(Softmax())
    train(seq)


def test_UNet():
    seq = tg.Sequential()
    model = UNet(input_channels=c, input_shape=(h, w))
    print('output channels:', model.output_channels)
    print('output shape:', model.output_shape)
    out_dim = np.prod(model.output_shape) * model.output_channels
    seq.add(model)
    seq.add(MaxPooling(poolsize=tuple(model.output_shape), stride=(1,1), padding='VALID'))
    seq.add(Flatten())
    seq.add(Linear(model.output_channels, nclass))
    seq.add(Softmax())
    train(seq)


if __name__ == '__main__':
    print('runtime test')
    test_VGG16()
    print('..VGG16 running test done')
    test_VGG19()
    print('..VGG19 running test done')
    test_ResNetSmall()
    print('..ResNetSmall running test done')
    test_ResNetBase()
    print('..ResNetBase running test done')
    test_DenseNet()
    print('..DenseNet running test done')
    test_UNet()
    print('..UNet running test done')
