# Menci
A distributed data-parallel framework based on TensorFlow

example
```python
model = Sequential()
model.add(Linear())
model.add(RELU())
model.add(Linear())
model.add(Softmax())

data = DataIterators()

distr = DistributedDataParrallel(model, data, gpus=[0,1,2])
distr.compile_train()
```
