


input_sequence = torch.tensor(dataset.scaled_data, dtype=torch.float32).unsqueeze(0)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted_price = model(input_sequence)  # Output shape: [1, 1]

# Denormalize the predicted price
predicted_price = scaler.inverse_transform(predicted_price.numpy())
print(f"Predicted Stock Price: ${predicted_price[0][0]:.2f}")