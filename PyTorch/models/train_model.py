import torch.optim as optim
from models.utils.nn import LSTMModel
from Dsets.training_dataset import Training_Dataset
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

class Training_Model():
    def __init__(self):
        inst = Training_Dataset()
        inst.create_data()
        self.X_train = torch.tensor(inst.X_train, dtype=torch.float32)  
        self.Y_train = torch.tensor(inst.Y_train, dtype=torch.float32)
        self.X_test = torch.tensor(inst.X_test, dtype=torch.float32)
        self.Y_test = torch.tensor(inst.Y_test, dtype=torch.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel().to(self.device)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)



    def train(self,num_epochs):
        print("Using device:", self.device)
        train_losses = []

        train_data = torch.utils.data.TensorDataset(self.X_train,self.Y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)

        for i in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_func(output, y)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss/len(train_loader)
            train_losses.append(avg)
            self.scheduler.step(avg)
            print(f"Epoch: {i+1} Loss: {loss.item():>4f}")

        plt.plot(range(num_epochs), train_losses)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def test(self):
        self.model.eval() 
        best = float('inf')
        with torch.no_grad():
            predictions = self.model(self.X_test)
            loss = self.loss_func(predictions, self.Y_test)
            changed = False
            if loss.item() < best or changed:
                print("Model updated")
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            print(f"Test Loss: {loss.item():.4f}")


if __name__ == "__main__":
    mp.set_start_method('fork')
    trnmdl = Training_Model()
    trnmdl.train(40)
    trnmdl.test()