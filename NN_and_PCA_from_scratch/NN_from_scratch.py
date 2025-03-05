
import random
import numpy as np
import scipy.io

'''
This file creates a neural network from scratch only using numpy, scipy, and random packages. 
'''

# sigmoid non-linearity
def sigmoid_forward(input):
    num = np.exp(-input)
    return 1 / (1 + num)


def sigmoid_backwards(input):
    return sigmoid_forward(input) * (1-sigmoid_forward(input))


# loss function
def mse_forward(input, labels):
    return np.mean(np.power(labels - input, 2))


def mse_backwards(input, labels):
    return 2 * (input - labels) / labels.size


# Implements the nonlinear layer (ie applies sigmoid function)
class SigmoidLayer():
    def __init__(self, sigmoid, sigmoid_prime):
        self.sigmoid = sigmoid
        self.sigmoid_prime = sigmoid_prime

    # returns the activated input
    def forward(self, input_data):
        self.input = input_data
        self.output = self.sigmoid(self.input)
        return self.output

    def backwards(self, output_error, learning_rate):
        return self.sigmoid_prime(self.input) * output_error


# linear (i.e. linear transformation) layer, wx + b
class Linear():
    def __init__(self, input_size, output_size, is_input=False):
        super(Linear, self).__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.is_input = is_input

    def forward(self, input):
        self.input = input
        # y = wx + b
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwards(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        if self.is_input:
            weights_error = np.dot(self.input.reshape(2,1), output_error)
        else:
            weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


## overall neural network class
class Network():
    def __init__(self):
        super(Network, self).__init__()
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, data):
        samples = len(data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result


# function for training the network for a given number of iterations
def train(model, data, labels, num_iterations, minibatch_size, learning_rate):
    samples = len(data)

    # shuffle data
    temp = list(zip(data, labels))
    random.shuffle(temp)
    x1, y1 = zip(*temp)
    x1, y1 = list(x1), list(y1)

    #training loop
    for i in range(num_iterations):
        err = 0
        batch = []
        batch_num = 0
        for j in range(samples):
            # forward propogation
            output = x1[j]
            for layer in model.layers:
                output = layer.forward(output)

            err += model.loss(y1[j], output)

            error = model.loss_prime(y1[j], output)
            batch.append(error)
            batch_num += 1
            if batch_num == minibatch_size:
                avg = 0
                for entry in batch:
                    avg += entry[0][0]
                avg /= len(batch)
                avg_error = np.array([[avg]])
                avg_error.reshape(1,1)
                for layer in reversed(model.layers):
                    avg_error = layer.backwards(avg_error, learning_rate)
                batch_num = 0
                batch.clear()

        err /= samples
        print('epoch %d/%d   error=%f' % (i + 1, num_iterations, err))


if __name__ == "__main__":
    # model 1
    # training data
    mat = scipy.io.loadmat('nn_data.mat')
    # normalize y1, y2
    x1 = mat['X1']
    y1 = mat['Y1']

    # this shows the image
    image1 = []
    for i in range(100):
        rowa = [0]*76
        image1.append(rowa)

    row = 0
    col = 0
    for pixel in y1:
        image1[row][col] = pixel
        col += 1
        if col > 75:
            row += 1
            col = 0
    # plt.imshow(image1, cmap='gray')
    # plt.show()

    # back to business
    y1 = [[x[0]/255] for x in y1]

    x2 = mat['X2']
    y2 = mat['Y2']

    y2 = [[x[0]/255, x[1]/255, x[2]/255] for x in y2]

    # # network
    net = Network()
    net.add(Linear(2, 180, is_input=True))
    net.add(SigmoidLayer(sigmoid_forward, sigmoid_backwards))
    net.add(Linear(180, 442))
    net.add(SigmoidLayer(sigmoid_forward, sigmoid_backwards))
    net.add(Linear(442, 1))
    net.add(SigmoidLayer(sigmoid_forward, sigmoid_backwards))

    # # train
    net.use(mse_forward, mse_backwards)
    train(net, x1, y1, 10, 10, 1E-2)

    out = net.predict(x1)
    out = [[x[0]*255] for x in out]

    # this shows the image
    image2 = []
    for i in range(100):
        rowb = [0] * 76
        image2.append(rowb)

    row = 0
    col = 0
    for pixel in out:
        image2[row][col] = pixel[0]
        col += 1
        if col > 75:
            row += 1
            col = 0
    # uncomment to show the image
    # plt.imshow(image2, cmap='gray')
    # plt.show()

    # model 2
    x2 = mat['X2']
    y2 = mat['Y2']

    # this shows the image
    image3 = []
    for i in range(133):
        rowa = [0] * 140
        image3.append(rowa)

    row = 0
    col = 0
    for pixel in y2:
        image3[row][col] = pixel
        col += 1
        if col > 139:
            row += 1
            col = 0
    # plt.imshow(image2)
    # plt.show()

    y2 = [[x[0] / 255, x[1] / 255, x[2] / 255] for x in y2]

    # # network
    net = Network()
    net.add(Linear(2, 180, is_input=True))
    net.add(SigmoidLayer(sigmoid_forward, sigmoid_backwards))
    net.add(Linear(180, 432))
    net.add(SigmoidLayer(sigmoid_forward, sigmoid_backwards))
    net.add(Linear(432, 3))
    net.add(SigmoidLayer(sigmoid_forward, sigmoid_backwards))

    # # train
    net.use(mse_forward, mse_backwards)
    train(net, x1, y1, 10, 10, 1E-2)

    out = net.predict(x1)
    out = [[x[0][0] * 255, x[0][1] * 255, x[0][2] * 255] for x in out]

    # this shows the image
    image4 = []
    for i in range(133):
        rowb = [0] * 140
        image4.append(rowb)

    row = 0
    col = 0
    for pixel in out:
        image4[row][col] = pixel[0]
        col += 1
        if col > 139:
            row += 1
            col = 0
    # uncomment to show the image
    # plt.imshow(image4)
    # plt.show()