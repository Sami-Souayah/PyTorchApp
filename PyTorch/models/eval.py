import torch
from nn import LSTMModel  
from Dsets.training_dataset import scaler
from Dsets.dataset import X_input


X_input = torch.tensor(X_input, dtype=torch.float32)

model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)

model.load_state_dict(torch.load("best_lstm_model.pth", weights_only=True))
model.eval() 

with torch.no_grad():
    predicted_price = model(X_input)

predicted_price = scaler.inverse_transform(predicted_price.detach().numpy())

print(f"Predicted stock price for the next day: {predicted_price[0][0]}")