import torch
from models.utils.nn import LSTMModel  
from Dsets.dataset import Dataset
import os


class Evaluate():
    def __init__(self, data):
        inst = Dataset(data)
        inst.create_input()
        self.scaler = inst.scaler
        self.X_input = inst.X_input
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel().to(self.device)
        self.weights = self.model.load_state_dict(torch.load('/Users/sami/Documents/Projects/PyTorchApp/PyTorch/best_lstm_model.pth', weights_only=True))
    
    def eval(self):
        print("Using device:", self.device)
        self.model.eval()
        with torch.no_grad():
            predicted_price = self.model(self.X_input)
        predicted_price = self.scaler.inverse_transform(predicted_price.detach().numpy())
        predicted_price = round(float(predicted_price[0][0]),2)
        return f"Predicted closing price for the next week: {predicted_price}"
    

