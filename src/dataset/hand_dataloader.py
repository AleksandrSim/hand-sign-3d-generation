import json

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class HandSignDataset(Dataset):
    def __init__(self, npz_file_path, json_file_path, normalize=False):
        # Load data
        npz_data = np.load(npz_file_path)
        self.data = npz_data['data']
        if normalize:
            self._normalize_data()

        # Load annotations and reverse them to get letters as keys
        with open(json_file_path, 'r') as json_file:
            self.annotations = {v: k for k, v in json.load(json_file).items()}

        self.dataset = self._create_dataset()

    def _normalize_data(self):

        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

    def _create_dataset(self):
        shortest_length = find_shortest_sequence_length(self.annotations)

        dataset = []
        letters = sorted(self.annotations, key=lambda x: int(self.annotations[x]))

        for i in range(len(letters) - 1):
            start_letter = letters[i]
            end_letter = letters[i + 1]
            start_frame = int(self.annotations[start_letter])
            end_frame = int(self.annotations[end_letter])
            sequence = self.data[:, :, start_frame:end_frame]

            downsampled_sequence = targeted_frame_removal(sequence, shortest_length, preserve_start_end_frames=5)

            dataset.append(((start_letter, end_letter), torch.tensor(downsampled_sequence, dtype=torch.float)))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def find_shortest_sequence_length(annotations):
    lengths = []
    sorted_frames = sorted(annotations.values(), key=lambda x: int(x))
    for i in range(len(sorted_frames) - 1):
        start_frame = int(sorted_frames[i])
        end_frame = int(sorted_frames[i + 1])
        length = end_frame - start_frame
        lengths.append(length)
    return min(lengths)

def calculate_downsampling_factor(target_length, sequence_length):
    factor = sequence_length / target_length
    return int(np.ceil(factor)) if factor > 1 else 1

def targeted_frame_removal(sequence, target_length, preserve_start_end_frames=5):
    current_length = sequence.shape[2]

    # Calculate the number of frames to preserve at the start and end
    preserve_count = 2 * preserve_start_end_frames

    if current_length > target_length and current_length > preserve_count:
        total_remove = current_length - target_length

        middle_remove_indices = np.linspace(preserve_start_end_frames, current_length - preserve_start_end_frames - 1, total_remove, dtype=int)
        mask = np.ones(current_length, dtype=bool)
        mask[middle_remove_indices] = False
        downsampled_sequence = sequence[:, :, mask]
        return downsampled_sequence

    return sequence  


if __name__ == '__main__':
    npz_file_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/npz/alphabet_new_100fps.fbx.npz'
    json_file_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/markup/alphabet_new_100fps.json'

    # Create the dataset
    dataset = HandSignDataset(npz_file_path, json_file_path, normalize=True)

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Example: Iterate over the train_loader
    for pair, sequence in train_loader:

        print(f"Letter pair: {pair}, Sequence shape: {sequence.shape}")