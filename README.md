# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Design Steps

### Step 1:

Import required libraries such as torch, torch.nn, torch.optim, numpy, pandas, and matplotlib.

### Step 2:
Load the dataset (e.g., stock closing prices from CSV), preprocess it by normalizing values between 0 and 1, and create input sequences for training/testing.

### Step 3:
Define the RNN model architecture with an input layer, hidden layers, and an output layer to predict stock prices.

### Step 4:
Compile the model using MSELoss as the loss function and Adam optimizer.

### Step 5:
Train the model on the training data, recording training losses for each epoch.

### Step 6:
Test the trained model on unseen data and visualize results by plotting the true stock prices vs. predicted stock prices.


## Program
#### Name: RIHAN AHAMED S
#### Register Number: 212224040276
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}")
    print("NAME: RIHAN AHAMED.S")
    print("reg.no: 212224040276")
```

## Output
<img width="1153" height="405" alt="image" src="https://github.com/user-attachments/assets/85a44042-57f2-478a-82ef-1f50093635b5" />

<img width="889" height="789" alt="image" src="https://github.com/user-attachments/assets/da6fc141-660d-42fb-9b04-4f8c585a3d65" />

<img width="1020" height="630" alt="image" src="https://github.com/user-attachments/assets/7461deb9-c7d7-42a3-a3da-63c2df3a45b5" />

<img width="1221" height="796" alt="image" src="https://github.com/user-attachments/assets/f174e0f0-8753-4c1c-9a6f-75de8918f037" />


### True Stock Price, Predicted Stock Price vs time

Include your plot here

### Predictions 

Predicted Price: [1095.8416]

Actual Price: [1115.65]

## Result

The RNN model was successfully implemented for stock price prediction.
