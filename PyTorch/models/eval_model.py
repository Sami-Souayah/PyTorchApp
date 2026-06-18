import torch
from models.utils.nn import LSTMModel  
from Dsets.evaldata import EvalData
import os
import matplotlib.pyplot as plt
import yfinance as yf


class Evaluate():
    def __init__(self, ticker):
        inst = EvalData(ticker)
        inst.find_ticker()
        inst.build_eval_features()
        inst.apply_scalers()
        inst.create_eval_sequence()
        inst.to_tensor()
        self.scaler = inst.price_scaler
        self.ticker = ticker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel().to(self.device)
        self.weights = self.model.load_state_dict(torch.load('best_lstm_model.pth', weights_only=True))
        self.X_input = inst.X_input.to(self.device)


    
    def eval(self):
        print("Using device:", self.device)
        self.model.eval()
        with torch.no_grad():
            predicted_price = self.model(self.X_input)
        predicted_price = self.scaler.inverse_transform(predicted_price.detach().cpu().numpy())
        predicted_price_flat = predicted_price.flatten()
        predicted_price_rounded = [round(float(p),2) for p in predicted_price_flat]


        print(f"Prices for {self.ticker}:",predicted_price_rounded)
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

        plt.title(f"Predicted Prices for the week for {self.ticker}")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    

