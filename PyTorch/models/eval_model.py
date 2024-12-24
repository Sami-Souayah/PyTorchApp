import torch
from models.utils.nn import LSTMModel  
from Dsets.training_dataset import Training_Dataset
from Dsets.dataset import Dataset


class Evaluate():
    def __init__(self):
        inst = Dataset()
        inst2 = Training_Dataset()
        self.scaler = inst2.scaler
        self.X_input = torch.tensor(inst.X_input, dtype=torch.float32)
        self.model = LSTMModel()
        self.weights = self.model.load_state_dict(torch.load("best_lstm_model.pth", weights_only=True))
    
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            predicted_price = self.model(self.X_input)
        predicted_price = self.scaler.inverse_transform(predicted_price.detach().numpy())
        return f"Predicted stock price for the next day: {predicted_price[0][0]}"