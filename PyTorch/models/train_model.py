import torch.optim as optim
from models.utils.nn import LSTMModel
import torch
from Dsets.transformations import Transformations
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

class Training_Model():
    def __init__(self):
        inst = Transformations()
        inst.load_data()
        inst.build_features()
        inst.split_data()
        inst.fit_scalers()
        inst.create_sequences()
        inst.to_tensor()
        self.best_loss = float("inf")
        self.X_train = inst.X_train
        self.Y_train = inst.Y_train
        self.X_test = inst.X_test
        self.Y_test = inst.Y_test
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
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_func(output, y)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss/len(train_loader)
            train_losses.append(avg)
            self.scheduler.step(avg)
            print(f"Epoch: {i+1} Loss: {avg:>4f}")

        plt.plot(range(num_epochs), train_losses)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def test(self):
        self.model.eval() 
        with torch.no_grad():
            predictions = self.model(self.X_test.to(self.device))
            loss = self.loss_func(predictions, self.Y_test.to(self.device))
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
                print("Model updated")
            print(f"Test Loss: {loss.item():.4f}")


if __name__ == "__main__":
    mp.set_start_method('fork')
    trnmdl = Training_Model()
    trnmdl.train(40)
    trnmdl.test()