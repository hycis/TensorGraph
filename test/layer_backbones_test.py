import tensorgraph as tg
from tensorgraph.layers.backbones import *
from tensorgraph.layers import Softmax, Flatten, Linear, MaxPooling, BaseModel, Concat, Select, NoChange
import tensorflow as tf
import os
from tensorgraph.trainobject import train as mytrain
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
        this_dir = os.path.dirname(os.path.realpath(__file__))
        writer = tf.summary.FileWriter(this_dir + '/tensorboard', sess.graph)
        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=1,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=1, randomize_split=False)
        writer.close()


def test_VGG16():
    seq = tg.Sequential()
    seq.add(VGG16())
    seq.add(Flatten())
    seq.add(Linear(this_dim=nclass))
    seq.add(Softmax())
    train(seq)


def test_VGG19():
    seq = tg.Sequential()
    seq.add(VGG19())
    seq.add(Flatten())
    seq.add(Linear(this_dim=nclass))
    seq.add(Softmax())
    train(seq)


def test_ResNetSmall():
    seq = tg.Sequential()
    seq.add(ResNetSmall(config=[1,1]))
    seq.add(MaxPooling(poolsize=(1,1), stride=(1,1), padding='VALID'))
    seq.add(Flatten())
    seq.add(Linear(this_dim=nclass))
    seq.add(Softmax())
    train(seq)


def test_ResNetBase():
    seq = tg.Sequential()
    seq.add(ResNetBase(config=[1,1,1,1]))
    seq.add(MaxPooling(poolsize=(1,1), stride=(1,1), padding='VALID'))
    seq.add(Flatten())
    seq.add(Linear(this_dim=nclass))
    seq.add(Softmax())
    train(seq)


def test_DenseNet():
    seq = tg.Sequential()
    seq.add(DenseNet(ndense=1, growth_rate=1, nlayer1blk=1))
    seq.add(MaxPooling(poolsize=(3,3), stride=(1,1), padding='VALID'))
    seq.add(Flatten())
    seq.add(Linear(this_dim=nclass))
    seq.add(Softmax())
    train(seq)


def test_UNet():
    seq = tg.Sequential()
    seq.add(UNet(input_shape=(h, w)))
    seq.add(MaxPooling(poolsize=(3,3), stride=(1,1), padding='VALID'))
    seq.add(Flatten())
    seq.add(Linear(this_dim=nclass))
    seq.add(Softmax())
    train(seq)


class XModel(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self):

        self.startnode = tg.StartNode(input_vars=[None])
        layers1 = []
        layers1.append(Linear(5))
        hn1 = tg.HiddenNode(prev=[self.startnode], input_merge_mode=Select(0), layers=layers1)

        layers2 = []
        layers2.append(Linear(8))
        hn2 = tg.HiddenNode(prev=[self.startnode], input_merge_mode=Select(1), layers=layers2)

        merge = tg.HiddenNode(prev=[hn1, hn2], input_merge_mode=Concat(axis=1))

        layers1a = []
        layers1a.append(Linear(20))
        hn1a = tg.HiddenNode(prev=[merge], layers=layers1a)

        layers2a = []
        layers2a.append(Linear(30))
        hn2a = tg.HiddenNode(prev=[merge], layers=layers2a)
        self.endnode = tg.EndNode(prev=[hn1a, hn2a])


def test_BaseModel():
    X1 = np.random.rand(10, 3).astype('float32')
    X2 = np.random.rand(10, 5).astype('float32')
    xmodel = XModel()
    y1, y2 = xmodel.train_fprop(X1, X2)
    assert y1.shape == (10, 20)
    assert y2.shape == (10, 30)


if __name__ == '__main__':
    # print('runtime test')
    # test_VGG16()
    # print('..VGG16 running test done')
    # test_VGG19()
    # print('..VGG19 running test done')
    # test_ResNetSmall()
    # print('..ResNetSmall running test done')
    # test_ResNetBase()
    # print('..ResNetBase running test done')
    # test_DenseNet()
    # print('..DenseNet running test done')
    # test_UNet()
    # print('..UNet running test done')
    test_BaseModel()
