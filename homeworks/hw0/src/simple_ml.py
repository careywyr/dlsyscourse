import struct
import numpy as np
import gzip

try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 打开图像文件
    with gzip.open(image_filename, 'rb') as img_file:
        # 读取文件头信息
        magic_number, num_images, rows, cols = np.frombuffer(img_file.read(16), dtype=np.uint32).byteswap()
        # 读取图像数据
        image_data = np.frombuffer(img_file.read(), dtype=np.uint8)
        # 将图像数据调整为 (num_images, rows * cols) 的形状并归一化
        X = image_data.reshape(num_images, rows * cols).astype(np.float32) / 255.0

    # 打开标签文件
    with gzip.open(label_filename, 'rb') as lbl_file:
        # 读取文件头信息
        magic_number, num_labels = np.frombuffer(lbl_file.read(8), dtype=np.uint32).byteswap()
        # 读取标签数据
        y = np.frombuffer(lbl_file.read(), dtype=np.uint8)
    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return np.mean(
        np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(Z.shape[0]), y]
    )
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples, input_dim = X.shape
    num_classes = theta.shape[1]

    for i in range(0, num_examples, batch):
        # 获取 mini-batch 的数据
        X_batch = X[i:i + batch]
        y_batch = y[i:i + batch]

        # Step 1: 计算 logits
        logits = X_batch @ theta  # shape (batch_size, num_classes)

        # Step 2: 计算 softmax 概率
        # 使用数值稳定的方式计算 softmax：先减去每行的最大值
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Step 3: 计算梯度
        # 创建一个形状为 (batch_size, num_classes) 的独热编码矩阵 I_y
        I_y = np.zeros_like(softmax_probs)
        I_y[np.arange(len(y_batch)), y_batch] = 1

        # 计算梯度: (input_dim, batch_size) @ (batch_size, num_classes) = (input_dim, num_classes)
        gradient = X_batch.T @ (softmax_probs - I_y) / len(y_batch)

        # Step 4: 更新 theta 参数 (in-place)
        theta -= lr * gradient
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples, input_dim = X.shape
    _, num_classes = W2.shape
    for i in range(0, num_examples, batch):
        X_batch = X[i:i + batch]
        y_batch = y[i:i + batch]

        # 1. 前向传播
        Z1 = np.maximum(0, X_batch @ W1)
        logits = Z1 @ W2

        # 2. 计算Softmax, 可以先减一下最大值防止数值溢出
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 3. 输入的y是标签值，转成one-hot的形式
        y_one_hot = np.zeros((y_batch.size, num_classes))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1

        # 4. 反向传播
        # 计算G2（相对于W2输出的梯度）
        G2 = (softmax_output - y_one_hot) / batch
        # G1同理，不过注意这里激活函数是ReLU，反向的时候可以写成(Z1 > 0)，输入值大于0导数才是1
        G1 = (G2 @ W2.T) * (Z1 > 0)

        # 计算梯度并更新W2
        grad_W2 = Z1.T @ G2
        W2 -= lr * grad_W2
        # 计算梯度并更新W1
        grad_W1 = X_batch.T @ G1
        W1 -= lr * grad_W1

    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
