import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import f1_score, accuracy_score


class Network:
    def __init__(self, epochs=100, learning_rate=0.001) -> None:
        (self.x_train, self.y_train_ori), (self.x_test, self.y_test_ori) = (
            mnist.load_data()
        )

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.nodes_n = 16
        self.outputs_n = 10
        self.mini_batch_size = 600

        self.initialize_variables()

    def initialize_variables(self):
        rand_range = [-1, 1]
        self.weights = [
            np.random.uniform(*rand_range, (self.nodes_n, 28 * 28)),
            np.random.uniform(*rand_range, (self.nodes_n, self.nodes_n)),
            np.random.uniform(*rand_range, (self.outputs_n, self.nodes_n)),
        ]
        self.biases = [
            np.zeros((self.nodes_n)),
            np.zeros((self.nodes_n)),
            np.zeros((self.outputs_n)),
        ]
        self.num_layers = len(self.weights) + 1

    def reshape_data(self, data):
        return data.reshape(len(data), -1)

    def standardizate_data(self, data):
        divisor = 255
        return [x / divisor for x in data]

    def one_hot_encoding(self, X):
        new_list = []
        for x in X:
            new_list.append([int(x == i) for i in range(self.outputs_n)])
        return new_list

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

    # @staticmethod
    # def sigmoid(z):
    #     return 1.0 / (1.0 + np.exp(-z))

    # def sigmoid_prime(self, z):
    #     """Derivative of the sigmoid function."""
    #     return self.sigmoid(z) * (1 - self.sigmoid(z))

    @staticmethod
    def softmax(z):
        e_x = np.exp(z - np.max(z))
        return e_x / np.sum(e_x)

    def softmax_derivative(self, z):
        return self.softmax(z) * (1 - self.softmax(z))

    def cross_entropy(self, A, Y, m):
        """cross-entropy"""
        Y = np.argmax(Y, axis=1)
        A = np.array(A)
        m = Y.shape[0]
        log_L = -np.log(A[range(m), Y])
        loss = np.sum(log_L) / m
        return loss

    def cross_entropy_derivative(self, A, y):
        y = np.argmax(y, axis=1)
        A = np.array(A)
        m = y.shape[0]
        A[range(m), y] -= 1
        grad = A / m
        return grad

    def feedforward(self, X):
        A = [X.copy()]
        for layer_i in range(self.num_layers - 1):
            z = (self.weights[layer_i] @ np.array(A[layer_i]).T).T + self.biases[
                layer_i
            ]
            if layer_i == self.num_layers - 2:
                A.append(self.softmax(z))
            else:
                A.append(self.relu(z))
        return A[-1]

    def backpropagation(self):
        costs = []
        for epoch in range(self.epochs):
            m = len(self.x_train)

            # feedforward
            A = [self.x_train]
            zs = []

            for layer_i in range(self.num_layers - 1):
                z = (self.weights[layer_i] @ np.array(A[layer_i]).T).T + self.biases[
                    layer_i
                ]
                zs.append(z)
                if layer_i == self.num_layers - 2:
                    A.append([self.softmax(z1) for z1 in z])
                else:
                    A.append(self.relu(z))

            costs.append(self.cross_entropy(A[-1], self.y_train, m))

            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], cost: {costs[-1]:.3f}")

            # backward pass
            dw = [np.zeros(w.shape) for w in self.weights]
            db = [np.zeros(b.shape) for b in self.biases]

            delta = self.cross_entropy_derivative(A[-1], self.y_train)

            dw[-1] = delta.T @ A[-2]
            db[-1] = np.sum(delta, axis=0)

            for layer_i in range(2, self.num_layers):
                delta = (
                    self.weights[-layer_i + 1].T @ delta.T
                ).T * self.relu_derivative(zs[-layer_i])

                dw[-layer_i] = delta.T @ A[-layer_i - 1]
                db[-layer_i] = np.sum(delta, axis=0)

            # update weights and biases
            self.weights = [
                w - self.learning_rate * d for w, d in zip(self.weights, dw)
            ]
            self.biases = [b - self.learning_rate * d for b, d in zip(self.biases, db)]

    def start_network(self):
        self.x_test = self.reshape_data(self.x_test)
        self.x_test = self.standardizate_data(self.x_test)

        self.x_train = self.reshape_data(self.x_train)
        self.x_train = self.standardizate_data(self.x_train)

        self.y_train = self.one_hot_encoding(self.y_train_ori)
        self.y_test = self.one_hot_encoding(self.y_test_ori)

        self.backpropagation()

    def show_images(self, num_images=10):

        for i in range(num_images):
            first_image = self.x_train[i]
            first_image = np.array(first_image, dtype="float")
            pixels = first_image.reshape((28, 28))
            plt.imshow(pixels, cmap="gray")
            plt.show()

    def predict(self, X):
        X = np.array(X)
        A = self.feedforward(X)

        return [np.argmax(a) for a in A]

    def test(self):
        Y_pred_train = self.predict(self.x_train)
        Y_pred_test = self.predict(self.x_test)

        print(
            "Train accuracy: {:.3f} %".format(
                accuracy_score(self.y_train_ori, Y_pred_train) * 100
            )
        )
        print(
            "Test accuracy: {:.3f} %".format(
                accuracy_score(self.y_test_ori, Y_pred_test) * 100
            )
        )

        f1_test = f1_score(self.y_test_ori, Y_pred_test, average="macro") * 100
        print(f"Test F1 score: {f1_test:.3f} %")


net = Network(epochs=100, learning_rate=0.01)

net.start_network()
net.test()
