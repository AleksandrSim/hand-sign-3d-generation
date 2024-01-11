import os
import sys
import json


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from src.process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS
from hand_dataloader import HandSignDataset
from torch.utils.data import DataLoader

class DataLoaderHandSignPlotter:
    def __init__(self, data_loader, speed=1):
        self.data_loader = data_loader
        self.speed = speed  # Speed parameter
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})
        plt.subplots_adjust(bottom=0.2)

        # Buttons
        self.axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(self.axnext, 'Next')
        self.bprev = Button(self.axprev, 'Previous')

        self.bnext.on_clicked(self.next_frame)
        self.bprev.on_clicked(self.prev_frame)

        # Load the first batch of data
        self.data_iter = iter(self.data_loader)
        self.load_next_batch()

    def load_next_batch(self):
        try:
            self.pair, self.sequence = next(self.data_iter)
            self.current_frame_idx = 0
            self.max_frame_idx = self.sequence.shape[3] - 1
        except StopIteration:
            print("End of DataLoader reached.")
            plt.close(self.fig)

    def plot_frame(self, frame_idx):
        frame_data = self.sequence[0, :, :, frame_idx].numpy()
        self.ax.clear()
        for start_bone, end_bone in HAND_BONES_CONNECTIONS:
            start_idx = HAND_BONES.index(start_bone)
            end_idx = HAND_BONES.index(end_bone)
            self.ax.plot([frame_data[start_idx, 0], frame_data[end_idx, 0]],
                         [frame_data[start_idx, 1], frame_data[end_idx, 1]],
                         [frame_data[start_idx, 2], frame_data[end_idx, 2]], 'ro-')
        self.ax.set_title(f"Letter pair: {self.pair[0]}, Frame: {frame_idx}")
        plt.draw()

    def next_frame(self, event):
        new_index = self.current_frame_idx + self.speed
        if new_index <= self.max_frame_idx:
            self.current_frame_idx = new_index
        else:
            self.load_next_batch()
        self.plot_frame(self.current_frame_idx)

    def prev_frame(self, event):
        new_index = self.current_frame_idx - self.speed
        if new_index >= 0:
            self.current_frame_idx = new_index
        self.plot_frame(self.current_frame_idx)

    def show(self):
        self.plot_frame(self.current_frame_idx)
        plt.show()

if __name__ == '__main__':
    npz_file_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/npz/alphabet_new_100fps.fbx.npz'
    json_file_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/markup/alphabet_new_100fps.json'

    # Create the dataset
    dataset = HandSignDataset(npz_file_path, json_file_path, normalize=True)

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Assuming you have already created a DataLoader named 'train_loader'
    plotter = DataLoaderHandSignPlotter(test_loader, speed=3)
    plotter.show()