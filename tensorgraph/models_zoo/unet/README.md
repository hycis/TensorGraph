# Model_zoo/Unet

An U-Net implementation in TensorGraph

## Train

You may refer training example in train.py

1. Prepared training data
2. Do necessary modification like filter size in model.py
3. Initialize model by import  tensorgraph.models_zoo.Unet.model.Unet
4. Define cost and optimizer
5. Feed data to tensorgraph.trainobject.train for training

## Test

You may refer testing example in test.py

1. Initialzie api class by import tensorgraph.models_zoo.Unet.unet_api
2. Define model_path to restore if there is any (same training data dims with testing)
3. Predict by feeding testing data
