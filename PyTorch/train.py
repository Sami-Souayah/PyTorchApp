import torch.optim as optim
import nn
from dataset import X_test,X_train,Y_test,Y_train
import torch


X_train = torch.tensor(X_train, dtype=torch.float32)  
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

model = nn.model

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(num_epochs, model, train_dataloader, loss_func):
    total_steps = len(train_dataloader)
    for i in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_func(output, Y_train)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {i+1} Loss: {loss.item():>4f}")