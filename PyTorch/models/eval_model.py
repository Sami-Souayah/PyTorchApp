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
        self.model = LSTMModel()
        model_path = os.path.join(os.path.dirname(__file__), "best_lstm_model.pth")
        self.weights = self.model.load_state_dict(torch.load('/Users/sami/Documents/Projects/PyTorchApp/PyTorch/best_lstm_model.pth', weights_only=True))
    
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            predicted_price = self.model(self.X_input)
        predicted_price = self.scaler.inverse_transform(predicted_price.detach().numpy())
        return f"Predicted closing price for the next day: {predicted_price[0][0]}"
    

Evaluate('RTX')