import torch.optim as optim
import nn as mdl
from dataset import X_train,Y_train, X_test, Y_test
import torch
import matplotlib.pyplot as plt



X_train = torch.tensor(X_train, dtype=torch.float32)  
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)  
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

model = mdl.model


loss_func = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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