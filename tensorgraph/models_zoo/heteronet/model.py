from ...node   import StartNode, HiddenNode, EndNode
from ...layers import BaseModel, Softmax, Graph, BatchNormalization
from ...layers import MaxPooling3D, RELU, Sum, Concat, Reshape
from .layers   import Conv3Dx

import tensorflow as tf

class Conv3DBlock(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, filters=32, kernel=(3,3,3)):
        self.startnode = StartNode([None])
        encode = []
        encode.append(Conv3Dx(num_filters=filters, kernel_size=kernel, stride=(1,1,1), padding='SAME'))
        encode.append(RELU())
        encode.append(Conv3Dx(num_filters=filters, kernel_size=kernel, stride=(1,1,1), padding='SAME'))
        encode.append(BatchNormalization())
        encode.append(RELU())
        out_hn         = HiddenNode(prev=[self.startnode], input_merge_mode=Sum(), layers=encode)
        self.endnode   = EndNode(prev=[out_hn])

class SingleEncoder(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self):
        self.startnode = StartNode([None])
        encode = []
        encode.append(Conv3DBlock(filters=32, kernel=(1,5,5)))
        encode.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='SAME'))
        encode.append(Conv3DBlock(filters=64, kernel=(1,5,5)))
        out_hn         = HiddenNode(prev=[self.startnode], input_merge_mode=Sum(), layers=encode)
        self.endnode   = EndNode(prev=[out_hn])

class MergeEncoder(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self):
        self.startnode = StartNode([None])
        encode2 = []
        # squeezing concat_conv_filters
        encode2.append(Conv3Dx(num_filters=64, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        encode2.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='SAME'))

        encode2.append(Conv3DBlock(filters=96, kernel=(2,3,3)))
        encode2.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='SAME'))
        encode2.append(Conv3DBlock(filters=128, kernel=(3,3,3)))
        encode2.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='SAME'))

        # fully_connected_layers, current_shape is (5, 6, 20, 20, 128)
        encode2.append(Conv3Dx(num_filters=500,kernel_size=(1,20,20), stride=(1,20,20), padding='SAME'))
        encode2.append(RELU())
        encode2.append(Conv3Dx(num_filters=125,kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        encode2.append(RELU())
        encode2.append(Conv3Dx(num_filters=75,kernel_size=(4,1,1), stride=(4,1,1), padding='SAME'))
        encode2.append(RELU())
        encode2.append(Conv3Dx(num_filters=30,kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        encode2.append(Reshape([-1,30]))
        encode2.append(Softmax())
        out_hn         = HiddenNode(prev=[self.startnode], input_merge_mode=Sum(), layers=encode2)
        self.endnode   = EndNode(prev=[out_hn])


class HeteroNet(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self):
        self.startnode1 = StartNode([None])
        self.startnode2 = StartNode([None])
        self.startnode3 = StartNode([None])
        layers1 = SingleEncoder()
        layers2 = MergeEncoder()
        t1_hn  = HiddenNode(prev=[self.startnode1],   input_merge_mode=Sum(),      layers=[layers1])
        t2_hn  = HiddenNode(prev=[self.startnode2],   input_merge_mode=Sum(),      layers=[layers1])
        tc_hn  = HiddenNode(prev=[self.startnode3],   input_merge_mode=Sum(),      layers=[layers1])
        out_hn = HiddenNode(prev=[t1_hn,t2_hn,tc_hn], input_merge_mode=Concat(-1), layers=[layers2])
        self.endnode = EndNode(prev=[out_hn])

    def _train_fprop(self, start1, start2, start3):
        self.startnode1.input_vars = [start1]
        self.startnode2.input_vars = [start2]
        self.startnode3.input_vars = [start3]
        graph = Graph(start=[self.startnode1,self.startnode2,self.startnode3], end=[self.endnode])
        return graph.train_fprop()

    def _test_fprop(self, start1, start2, start3):
        self.startnode1.input_vars = [start1]
        self.startnode2.input_vars = [start2]
        self.startnode3.input_vars = [start3]
        graph = Graph(start=[self.startnode1,self.startnode2,self.startnode3], end=[self.endnode])
        return graph.test_fprop()



if __name__ == '__main__':
    import os
    tf.reset_default_graph()
    X_ph = tf.placeholder(tf.float32, [5,4,320,320,1])
    NN   = HeteroNet()
    out  = NN.train_fprop(X_ph,X_ph,X_ph)[0]
    print( tf.global_variables() )
    with tf.Session() as sess:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        writer = tf.summary.FileWriter(this_dir + '/tensorboard', sess.graph)
        sess.close()
