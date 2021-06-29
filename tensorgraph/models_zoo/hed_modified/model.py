
from ...node import StartNode, HiddenNode, EndNode
from ...layers import BaseLayer, BaseModel, Conv3D, Conv3D_Transpose, MaxPooling3D, \
                      ELU, Sigmoid, Dropout, Concat, BatchNormalization

class HED_Modified(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, channels, side_features, output_shape=(96,224,224), output_channels=3, droprate=0.03):

        layers_1 = []
        side_1   = []
        layers_2 = []
        side_2   = []
        layers_3 = []
        side_3   = []
        layers_4 = []
        side_4   = []
        layers_5 = []
        side_5   = []
        layers_fuse = []

        self.startnode = StartNode(input_vars=[None])

        # Block 1 #
        layers_1.append(Conv3D(num_filters=channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_1.append(BatchNormalization())
        layers_1.append(ELU())
        layers_1.append(Conv3D(num_filters=channels, kernel_size=(1,3,3), stride=(1,1,1), padding='SAME'))
        layers_1.append(BatchNormalization())
        layers_1.append(ELU())
        h1 = HiddenNode(prev=[self.startnode], layers=layers_1)

        side_1.append(Conv3D(num_filters=side_features, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        side_1.append(Conv3D_Transpose(num_filters=side_features, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        s1 = HiddenNode(prev=[h1], layers=side_1)

        layers_2.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='VALID'))
        layers_2.append(Dropout(dropout_below=droprate))


        # Block 2 #
        layers_2.append(Conv3D(num_filters=2*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_2.append(BatchNormalization())
        layers_2.append(ELU())
        layers_2.append(Conv3D(num_filters=2*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_2.append(BatchNormalization())
        layers_2.append(ELU())
        h2 = HiddenNode(prev=[h1], layers=layers_2)

        side_2.append(Conv3D(num_filters=side_features, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        side_2.append(Conv3D_Transpose(num_filters=side_features, kernel_size=(1,4,4), stride=(1,2,2), padding='SAME'))
        s2 = HiddenNode(prev=[h2], layers=side_2)

        layers_3.append(MaxPooling3D(poolsize=(2,2,2), stride=(2,2,2), padding='VALID'))
        layers_3.append(Dropout(dropout_below=droprate))


        # Block 3 #
        layers_3.append(Conv3D(num_filters=4*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_3.append(BatchNormalization())
        layers_3.append(ELU())
        layers_3.append(Conv3D(num_filters=4*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_3.append(BatchNormalization())
        layers_3.append(ELU())
        layers_3.append(Conv3D(num_filters=4*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_3.append(BatchNormalization())
        layers_3.append(ELU())
        h3 = HiddenNode(prev=[h2], layers=layers_3)

        side_3.append(Conv3D(num_filters=side_features, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        side_3.append(Conv3D_Transpose(num_filters=side_features, kernel_size=(4,8,8), stride=(2,4,4), padding='SAME'))
        s3 = HiddenNode(prev=[h3], layers=side_3)

        layers_4.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='VALID'))
        layers_4.append(Dropout(dropout_below=droprate))


        # Block 4 #
        layers_4.append(Conv3D(num_filters=8*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_4.append(BatchNormalization())
        layers_4.append(ELU())
        layers_4.append(Conv3D(num_filters=8*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_4.append(BatchNormalization())
        layers_4.append(ELU())
        layers_4.append(Conv3D(num_filters=8*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_4.append(BatchNormalization())
        layers_4.append(ELU())
        h4 = HiddenNode(prev=[h3], layers=layers_4)

        side_4.append(Conv3D(num_filters=side_features, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        side_4.append(Conv3D_Transpose(num_filters=side_features, kernel_size=(4,16,16), stride=(2,8,8), padding='SAME'))
        s4 = HiddenNode(prev=[h4], layers=side_4)

        layers_5.append(MaxPooling3D(poolsize=(2,2,2), stride=(2,2,2), padding='VALID'))
        layers_5.append(Dropout(dropout_below=droprate))

        # Block 5 #
        layers_5.append(Conv3D(num_filters=16*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_5.append(BatchNormalization())
        layers_5.append(ELU())
        layers_5.append(Conv3D(num_filters=16*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_5.append(BatchNormalization())
        layers_5.append(ELU())
        layers_5.append(Conv3D(num_filters=16*channels, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        layers_5.append(BatchNormalization())
        layers_5.append(ELU())
        layers_5.append(Dropout(dropout_below=droprate))
        h5 = HiddenNode(prev=[h4], layers=layers_5)

        side_5.append(Conv3D(num_filters=side_features, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))
        side_5.append(Conv3D_Transpose(num_filters=side_features, kernel_size=(8,32,32), stride=(4,16,16), padding='SAME'))
        s5 = HiddenNode(prev=[h5], layers=side_5)


        layers_fuse.append(Conv3D(num_filters=output_channels, kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))

        logits = HiddenNode(prev=[s1, s2, s3, s4, s5],
                            input_merge_mode=Concat(axis=-1),
                            layers=layers_fuse)
        y_hat = HiddenNode(prev=[logits],
                           layers=[Sigmoid()])

        self.endnode = EndNode(prev=[logits, y_hat])
