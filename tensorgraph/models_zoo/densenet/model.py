
from ...node import StartNode, HiddenNode, EndNode
from ...layers import BaseModel, DenseNet, MaxPooling, Flatten, Linear, Softmax


class MyDenseNet(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nclass):
        layers = []
        layers.append(DenseNet(ndense=1, growth_rate=1, nlayer1blk=1))
        layers.append(MaxPooling(poolsize=(3,3), stride=(1,1), padding='VALID'))
        layers.append(Flatten())
        layers.append(Linear(this_dim=nclass))
        layers.append(Softmax())
        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])
