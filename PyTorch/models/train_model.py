import torch.optim as optim
from models.utils.nn import LSTMModel
from Dsets.training_dataset import Training_Dataset
import torch
import matplotlib.pyplot as plt
import os

class Training_Model():
    def __init__(self):
        inst = Training_Dataset()
        inst.create_data()
        self.X_train = torch.tensor(inst.X_train, dtype=torch.float32)  
        self.Y_train = torch.tensor(inst.Y_train, dtype=torch.float32)
        self.X_test = torch.tensor(inst.X_test, dtype=torch.float32)
        self.Y_test = torch.tensor(inst.Y_test, dtype=torch.float32)
        self.model = LSTMModel()
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


    def train(self,num_epochs):
        for i in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = self.loss_func(output, self.Y_train)

            loss.backward()
            self.optimizer.step()
            print(f"Epoch: {i+1} Loss: {loss.item():>4f}")

    def test(self):
        self.model.eval() 
        with torch.no_grad():
            predictions = self.model(self.X_test)
            loss = self.loss_func(predictions, self.Y_test)
            torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            print(f"Test Loss: {loss.item():.4f}")
    
    def graph_test(self):
        predictions = self.model(self.X_test).detach().numpy()
        actual = self.Y_test.numpy()
        plt.plot(predictions, label="Predicted")
        plt.plot(actual, label="Actual")
        plt.legend()
        plt.show()
