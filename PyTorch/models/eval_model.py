import torch
from models.utils.nn import LSTMModel  
from Dsets.dataset import Dataset
import os
import matplotlib.pyplot as plt

class Evaluate():
    def __init__(self, data):
        self.inst = Dataset(data)
        self.inst.create_input()
        self.scaler = self.inst.price_scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_input = self.inst.X_input.to(self.device)
        self.model = LSTMModel().to(self.device)
        self.weights = self.model.load_state_dict(torch.load('/Users/sami/Documents/Projects/PyTorchApp/PyTorch/best_lstm_model.pth', weights_only=True))
    
    def eval(self):
        print("Using device:", self.device)
        self.model.eval()
        with torch.no_grad():
            predicted_price = self.model(self.X_input)
        predicted_price = self.scaler.inverse_transform(predicted_price.detach().cpu().numpy())
        predicted_price_flat = predicted_price.flatten()
        predicted_price_rounded = [round(float(p),2) for p in predicted_price_flat]

        last_week_prices = predicted_price_rounded[-7:]

        print(f"Prices for {self.inst.name}:",predicted_price_rounded)
        days = list(range(1, len(predicted_price_rounded) + 1))
        
        
        plt.figure(figsize=(16, 8))
        plt.plot(days, predicted_price_rounded, label="Predicted Prices", color="blue")

        plt.scatter(days, predicted_price_rounded, color="red")

        for i, price in enumerate(predicted_price_rounded):
            plt.text(
            days[i],
            predicted_price_rounded[i] + 0.01 * max(predicted_price_rounded),  
            f"{price}",
            ha="center",
            fontsize=8,
            color="black",
        )
        plt.ylim(min(predicted_price_rounded) * 0.95, max(predicted_price_rounded) * 1.05)

        plt.title(f"Predicted Prices for the week for {self.inst.name}")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    

