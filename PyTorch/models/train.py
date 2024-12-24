import torch.optim as optim
import nn as mdl
from Dsets.training_dataset import Training_Dataset
import torch
import matplotlib.pyplot as plt




class Training_Model():
    def __init__(self):
        inst = Training_Dataset()
        self.X_train = torch.tensor(inst.X_train, dtype=torch.float32)  
        self.Y_train = torch.tensor(inst.Y_train, dtype=torch.float32).unsqueeze(1)  
        self.X_test = torch.tensor(inst.X_test, dtype=torch.float32)
        self.Y_test = torch.tensor(inst.Y_test, dtype=torch.float32).unsqueeze(1)
        self.model = mdl.model
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


    def train(num_epochs, model, loss_func, optim):
        for i in range(num_epochs):
            model.train()
            optim.zero_grad()
            output = model(X_train)
            loss = loss_func(output, Y_train)

            loss.backward()
            optimizer.step()
            print(f"Epoch: {i+1} Loss: {loss.item():>4f}")

def evaluate(model, X_test, Y_test, loss_func):
    model.eval() 
    with torch.no_grad():
        predictions = model(X_test)
        loss = loss_func(predictions, Y_test)
        print(f"Test Loss: {loss.item():.4f}")


train(250, model, loss_func, optimizer)

evaluate(model,X_test,Y_test,loss_func)

torch.save(model.state_dict(), "best_lstm_model.pth")

predictions = model(X_test).detach().numpy()
actual = Y_test.numpy()

plt.plot(predictions, label="Predicted")
plt.plot(actual, label="Actual")
plt.legend()
plt.show()