import torch
import torch.nn as nn

class GestureSmoothingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GestureSmoothingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for smoothing
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Applying the fully connected layer to each time step
        out = self.fc(out)
        return out

if __name__ == '__main__':
    # Define parameters
    input_size = 3  # XYZ coordinates
    hidden_size = 128  # number of features in the hidden state
    num_layers = 2  # number of stacked LSTM layers
    output_size = 3  # output size (XYZ coordinates after smoothing)

    # Create a model instance
    model = GestureSmoothingLSTM(input_size, hidden_size, num_layers, output_size)

    # Generate a dummy input (e.g., batch_size=1, sequence_length=10, input_size=3)
    dummy_input = torch.randn(1, 10, input_size)

    # Forward pass
    output = model(dummy_input)

    # Print output shape
    print("Output Shape:", output.shape)
