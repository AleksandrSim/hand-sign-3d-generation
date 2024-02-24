import torch
import torch.nn as nn

from src.process_data.utils import letter_to_index, coordinates_input_gt, letter_to_index


class GestureSmoothingLSTM(nn.Module):
    def __init__(self, num_letters, letter_embedding_dim, hidden_size, num_layers, keypoints, coords, sequence_length):
        super(GestureSmoothingLSTM, self).__init__()
        self.letter_embedding = nn.Embedding(num_letters, letter_embedding_dim)
        total_input_size = letter_embedding_dim * 2 + keypoints * coords * 2
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, keypoints * coords)
        self.sequence_length = sequence_length
        self.keypoints = keypoints
        self.coords = coords
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, start_letters, end_letters, start_coords, end_coords):
        # Embed the start and end letters
        start_embeds = self.letter_embedding(start_letters)
        end_embeds = self.letter_embedding(end_letters)

        # Concatenate embeddings with coordinates
        combined = torch.cat((start_embeds, end_embeds, start_coords, end_coords), dim=-1)

        h0 = torch.zeros(self.num_layers, combined.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, combined.size(0), self.hidden_size)

        outputs = torch.zeros(combined.size(0), self.keypoints, self.coords, self.sequence_length)

        for t in range(self.sequence_length):
            combined_timestep = combined.unsqueeze(1)  # Add sequence dimension
            lstm_out, (h0, c0) = self.lstm(combined_timestep, (h0, c0))
            fc_out = self.fc(lstm_out.squeeze(1))
            outputs[:, :, :, t] = fc_out.view(-1, self.keypoints, self.coords)
        return outputs
    
    
# Example Usage
if __name__ == '__main__':
    # Define parameters
    num_letters = len(letter_to_index)  # Total number of unique letters
    letter_embedding_dim = 10  # Size of the letter embedding
    hidden_size = 128  # Number of features in the hidden state
    num_layers = 2  # Number of stacked LSTM layers
    keypoints = 19  # Number of keypoints
    coords = 3  # Number of coordinates (XYZ)
    sequence_length = 111  # Length of the sequence to generate

    # Create the model
    model = GestureSmoothingLSTM(num_letters, letter_embedding_dim, hidden_size, num_layers, keypoints, coords, sequence_length)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Test inference speed
    import time

    start_letters = torch.tensor([letter_to_index['A']], dtype=torch.long)  # Shape: [1]
    end_letters = torch.tensor([letter_to_index['V']], dtype=torch.long)  # Shape: [1]

    # Reshape your coordinates and ensure they are float tensors
    start_coords = torch.tensor(coordinates_input_gt['A'].reshape(-1), dtype=torch.float32).unsqueeze(0)  # Shape: [1, 57]
    end_coords = torch.tensor(coordinates_input_gt['B'].reshape(-1), dtype=torch.float32).unsqueeze(0)  
        
    print(end_coords.shape)
    start_time = time.time()
    output = model(start_letters, end_letters, start_coords, end_coords)
    end_time = time.time()
    # Print output shape
    inference_time = end_time - start_time

    print(total_params), print(total_trainable_params), print(inference_time)
    print("Output Shape:", output.shape)
