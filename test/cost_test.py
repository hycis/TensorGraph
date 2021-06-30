import tensorgraph as tg
import tensorflow as tf
import numpy as np
from tensorgraph.utils import make_one_hot
from sklearn.metrics import f1_score

def test_binary_f1():
    ph1 = tf.placeholder('int32', [None, 2])
    ph2 = tf.placeholder('int32', [None, 2])

    f1_sb = tg.cost.binary_f1(ph1, ph2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y1 = np.random.randint(0, 2, 100)
        y2 = np.random.randint(0 ,2, 100)
        sc1 = f1_score(y1, y2)

        y1_oh = make_one_hot(y1, 2)
        y2_oh = make_one_hot(y2, 2)
        sc2 = sess.run(f1_sb, feed_dict={ph1:y1_oh, ph2:y2_oh})
    assert (sc1 - sc2)**2 < 1e-6
    # assert sc1 == sc2


def test_image_f1():
    ph1 = tf.placeholder('int32', [None, 3, 4, 5])
    ph2 = tf.placeholder('int32', [None, 3, 4, 5])

    f1_sb = tg.cost.image_f1(ph1, ph2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y1 = np.random.random_integers(0, 1, [10, 3, 4, 5])
        y2 = np.random.random_integers(0 ,1, [10, 3, 4, 5])
        print(sess.run(f1_sb, feed_dict={ph1:y1, ph2:y2}))


# if __name__ == '__main__':
#     test_image_f1()
