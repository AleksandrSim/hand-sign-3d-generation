import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.hand_sequence import GestureSmoothingLSTM
from src.dataset.hand_dataloader import HandSignDataset
from src.process_data.utils import coordinates_input_gt, letter_to_index

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=1,
                learning_rate=0.0001, save_path=None, load_path=None):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.save_path = save_path
        self.load_path = load_path

    def before_train(self):
        self.model.train()

    def one_epoch_train(self, epoch):
        running_loss = 0.0
        for i, data in enumerate(self.train_loader, 0):
            letters, targets = data
            start_letters, end_letters = letters[0][0], letters[1][0]

            # Convert letters to indices and prepare coordinate tensors
            start_letter_idx = torch.tensor([letter_to_index[start_letters]], dtype=torch.long)
            end_letter_idx = torch.tensor([letter_to_index[end_letters]], dtype=torch.long)
            start_coords = torch.tensor(coordinates_input_gt[start_letters].reshape(-1), dtype=torch.float32).unsqueeze(0)
            end_coords = torch.tensor(coordinates_input_gt[end_letters].reshape(-1), dtype=torch.float32).unsqueeze(0)


            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(start_letter_idx, end_letter_idx, start_coords, end_coords)

            # Calculate loss and backpropagate
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Update running loss
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss / 10:.6f}')
                running_loss = 0.0

    def validate(self):
        if self.val_loader is None:
            print("No validation dataset provided.")
            return
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                start_letters, end_letters, start_coords, end_coords, targets = data
                outputs = self.model(start_letters, end_letters, start_coords, end_coords)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f'Validation Loss: {avg_loss:.4f}')

    def save_checkpoint(self, epoch, path):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint['epoch']

    def train(self, num_epochs, checkpoint_path=None):
        start_epoch = 0
        if checkpoint_path is not None:
            start_epoch = self.load_checkpoint(checkpoint_path)

        for epoch in range(start_epoch, num_epochs):
            self.before_train()
            self.one_epoch_train(epoch)
            if self.val_loader:
                self.validate()

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_name = f'lstm_fixed_start_epoch_{epoch}.pth'
                self.save_checkpoint(epoch, os.path.join(self.save_path, checkpoint_name))
                print(f'Epoch {epoch} checkpoint saved')


if __name__ == '__main__':
    npz_file_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/npz/alphabet_new_100fps.fbx.npz'
    json_file_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/markup/alphabet_new_100fps.json'

    # Create the dataset
    dataset = HandSignDataset(npz_file_path, json_file_path, normalize=True)

    num_letters = len(letter_to_index)

    letter_embedding_dim = 10  
    hidden_size = 128  
    num_layers = 2  
    keypoints = 19  
    coords = 3  
    sequence_length = 111  

    model = GestureSmoothingLSTM(num_letters, letter_embedding_dim, hidden_size, 
                                 num_layers, keypoints, coords, sequence_length)

    save_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/model_weights'
    # Assuming train_dataset and val_dataset are defined
    trainer = Trainer(model, dataset,save_path=save_path)
    trainer.train(num_epochs=500)
