
from tensorgraph.dataset import Mnist


def mnist_test():
    X_train, y_train, X_test, y_test = Mnist(binary=True, onehot=False, flatten=False)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


if __name__ == '__main__':
    mnist_test()
