import torch
from torch import nn
from sklearn import datasets
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, input_num=28 * 28, h1=16, h2=16, output_num=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_num, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_num),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


class EvaluateModel:
    def __init__(self, epochs=1000, lr=0.01) -> None:
        self.model = Model().to("cuda")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def load_data(self) -> None:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.int64)
        self.x_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

        self.y_train = nn.functional.one_hot(self.y_train, num_classes=10).float()

    def train_model(self) -> None:
        with torch.cuda.device(0):
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                outputs = self.model(self.x_train.cuda())
                loss = self.criterion(outputs, self.y_train.cuda())
                loss.backward()
                self.optimizer.step()
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

    def predict(self, X):
        pred = self.model(X.cuda())
        _, pred = torch.max(pred, 1)
        return pred

    def test_model(self) -> None:
        predicted_test = self.predict(self.x_test)

        accuracy_test = accuracy_score(
            self.y_test.cpu().numpy(), predicted_test.cpu().numpy()
        )
        f1_test = f1_score(
            self.y_test.cpu().numpy(), predicted_test.cpu().numpy(), average="macro"
        )

        print(f"Test Accuracy: {accuracy_test*100:.2f} %")
        print(f"Test F1 Score: {f1_test*100:.2f} %")

    def save_model_to_file(self, filename="digits_recognition_model.pt") -> None:
        while True:
            data = input("Do you want to save this model? (Y/N): ").lower()
            if data not in ("y", "n"):
                print("Not an appropriate choice.")
            elif data == "y":
                torch.save(self.model.state_dict(), filename)
                print("Model saved")
                break
            else:
                break

    def load_model(self, filename="digits_recognition_model.pt"):
        while True:
            data = input(
                f'Do you want to load model from "{filename}"? (Y/N): '
            ).lower()
            if data not in ("y", "n"):
                print("Not an appropriate choice.")
            elif data == "y":
                try:
                    self.model.load_state_dict(torch.load(filename))
                except FileNotFoundError:
                    print(f'File "{filename}" with model not found')
                except:
                    print("An exception occurred while loading model from file")
                print("Model loaded")
                return data
            else:
                return data

    def start(self) -> None:
        self.load_data()

        decision = self.load_model()

        if decision == "n":
            print("Starting training new model")
            self.train_model()

        self.test_model()

        if decision == "n":
            self.save_model_to_file()


ev1 = EvaluateModel()
ev1.start()
