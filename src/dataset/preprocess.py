import os
import sys
import json
                                            
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from src.process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS


class HandSignData:
    def __init__(self, file_path, annotations):
        self.data = np.load(file_path)['data']
        self.annotations = self.load_annotations(annotations)
        self.frames = self._extract_frames()

    def load_annotations(self, annotations):
        with open(annotations, 'r') as f:
            return json.load(f)

    def _extract_frames(self):
        return {ltr: frame for frame, ltr in self.annotations.items()}

    def get_frame(self, letter):
        frame_idx = int(self.frames.get(letter))
        if frame_idx is not None:
            return self.data[:, :, frame_idx]
        return None

    def get_frame_by_index(self, index):
        if index is not None and 0 <= index < self.data.shape[2]:
            return self.data[:, :, index]
        return None
    

class HandSignPlotter:
    def __init__(self, data_obj, start_letter, end_letter, show_range=False, speed=1):
        self.data_obj = data_obj
        self.show_range = show_range
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

        # Initialize frame range
        self.start_idx = int(self.data_obj.frames[start_letter])
        self.end_idx = int(self.data_obj.frames[end_letter])
        self.current_frame_idx = self.start_idx

        if self.show_range:
            self.frame_range = range(self.start_idx, self.end_idx + 1)
        else:
            self.frame_range = [self.start_idx]

    def plot_frame(self, frame_idx):
        frame_data = self.data_obj.get_frame_by_index(frame_idx)
        if frame_data is not None:
            self.ax.clear()
            for start_bone, end_bone in HAND_BONES_CONNECTIONS:
                start_idx = HAND_BONES.index(start_bone)
                end_idx = HAND_BONES.index(end_bone)
                self.ax.plot([frame_data[start_idx, 0], frame_data[end_idx, 0]],
                             [frame_data[start_idx, 1], frame_data[end_idx, 1]],
                             [frame_data[start_idx, 2], frame_data[end_idx, 2]], 'ro-')
            self.ax.set_title(f"Frame: {frame_idx}")
            plt.draw()

    def next_frame(self, event):
        new_index = self.current_frame_idx + self.speed
        if new_index <= self.end_idx:
            self.current_frame_idx = new_index
        self.plot_frame(self.current_frame_idx)

    def prev_frame(self, event):
        new_index = self.current_frame_idx - self.speed
        if new_index >= self.start_idx:
            self.current_frame_idx = new_index
        self.plot_frame(self.current_frame_idx)

    def show(self):
        self.plot_frame(self.current_frame_idx)
        plt.show()


if __name__ == '__main__':
    file_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/npz/alphabet_new_100fps.fbx.npz'
    annotations = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/markup/alphabet_new_100fps.json'

    data_obj = HandSignData(file_path, annotations)

    # Specify the start and end letters, and whether to show the range
    start_letter = 'A'
    end_letter = 'B'
    show_range = True  # Set to False to show only the start letter

    plotter = HandSignPlotter(data_obj, start_letter, end_letter, show_range=True, speed=10)
    plotter.show()