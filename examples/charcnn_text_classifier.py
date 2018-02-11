
import tensorflow as tf
import tensorgraph as tg
from tensorgraph.layers import Reshape, Embedding, Conv2D, RELU, Linear, Flatten, ReduceSum, Softmax
from nltk.tokenize import RegexpTokenizer
from nlpbox import CharNumberEncoder, CatNumberEncoder
from tensorgraph.utils import valid, split_df, make_one_hot
from tensorgraph.cost import entropy, accuracy
import pandas
import numpy as np

# character CNN
def model(word_len, sent_len, nclass):
    unicode_size = 1000
    ch_embed_dim = 20

    h, w = valid(ch_embed_dim, word_len, stride=(1,1), kernel_size=(ch_embed_dim,5))
    h, w = valid(h, w, stride=(1,1), kernel_size=(1,5))
    h, w = valid(h, w, stride=(1,2), kernel_size=(1,5))
    conv_out_dim = int(h * w * 60)

    X_ph = tf.placeholder('int32', [None, sent_len, word_len])
    input_sn = tg.StartNode(input_vars=[X_ph])
    charcnn_hn = tg.HiddenNode(prev=[input_sn],
                               layers=[Reshape(shape=(-1, word_len)),
                                    Embedding(cat_dim=unicode_size,
                                              encode_dim=ch_embed_dim,
                                              zero_pad=True),
                                    Reshape(shape=(-1, ch_embed_dim, word_len, 1)),
                                    Conv2D(input_channels=1, num_filters=20, padding='VALID',
                                           kernel_size=(ch_embed_dim,5), stride=(1,1)),
                                    RELU(),
                                    Conv2D(input_channels=20, num_filters=40, padding='VALID',
                                           kernel_size=(1,5), stride=(1,1)),
                                    RELU(),
                                    Conv2D(input_channels=40, num_filters=60, padding='VALID',
                                           kernel_size=(1,5), stride=(1,2)),
                                    RELU(),
                                    Flatten(),
                                    Linear(conv_out_dim, nclass),
                                    Reshape((-1, sent_len, nclass)),
                                    ReduceSum(1),
                                    Softmax()
                                    ])

    output_en = tg.EndNode(prev=[charcnn_hn])
    graph = tg.Graph(start=[input_sn], end=[output_en])
    y_train_sb = graph.train_fprop()[0]
    y_test_sb = graph.test_fprop()[0]

    return X_ph, y_train_sb, y_test_sb


def tweets(word_len, sent_len, train_valid_ratio=[5,1]):
    df = pandas.read_csv('tweets_large.csv')
    field = 'text'
    label = 'label'
    tokenizer = RegexpTokenizer(r'\w+')

    # encode characters into numbers
    encoder = CharNumberEncoder(df[field].values, tokenizer=tokenizer,
                                word_len=word_len, sent_len=sent_len)
    encoder.build_char_map()
    encode_X = encoder.make_char_embed()

    # encode categories into one hot array
    cat_encoder = CatNumberEncoder(df[label])
    cat_encoder.build_cat_map()

    encode_y = cat_encoder.make_cat_embed()
    nclass = len(np.unique(encode_y))
    encode_y = make_one_hot(encode_y, nclass)

    return encode_X, encode_y, nclass


def train():
    from tensorgraph.trainobject import train as mytrain
    with tf.Session() as sess:
        word_len = 20
        sent_len = 50

        # load data
        X_train, y_train, nclass = tweets(word_len, sent_len)

        # build model
        X_ph, y_train_sb, y_test_sb = model(word_len, sent_len, nclass)
        y_ph = tf.placeholder('float32', [None, nclass])

        # set cost and optimizer
        train_cost_sb = entropy(y_ph, y_train_sb)
        optimizer = tf.train.AdamOptimizer(0.001)
        test_accu_sb = accuracy(y_ph, y_test_sb)

        # train model
        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=100,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=64, randomize_split=False)


if __name__ == '__main__':
    train()
