import os
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.process_data.aggregated_npz import char_index_map

from src.process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS
HAND_BONES_INDEXES = list(range(19))


char_index_map = {
    'A': 0, 'B': 1, 'CH': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'HARD': 8, 'I': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'R': 16, 'S': 17, 'SH': 18,
    'SHCH': 19, 'SOFT': 20, 'T': 21, 'TS': 22, 'U': 23, 'V': 24, 'Y': 25, 'YA': 26,
    'YI': 27, 'YO': 28, 'YU': 29, 'Z': 30, 'ZH': 31
}

# Assuming HAND_BONES, HAND_BONES_CONNECTIONS, and char_index_map are defined as before

class HandTransitionVisualizer:
    def __init__(self, npz_path, word, visualize=True):
        self.word = word
        self.visualize = visualize
        self.data = self.load_data(npz_path)
        self.transitions_data = self.get_transitions_data()
        if visualize:
            self.current_transition = 0
            self.current_frame = 0
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.gcf().canvas.mpl_connect('key_press_event', self.on_key)

    def load_data(self, npz_path):
        return np.load(npz_path)['data']

    def get_transitions_data(self):
        transitions_data = []
        for i in range(len(self.word) - 1):
            start_char = self.word[i]
            end_char = self.word[i + 1]
            if start_char in char_index_map and end_char in char_index_map:
                start_index = char_index_map[start_char]
                end_index = char_index_map[end_char]
                transition_data = self.data[start_index, end_index, :, :, :]
                non_zero_frames_mask = np.any(transition_data != 0, axis=(0, 1))
                if non_zero_frames_mask.size > 0:
                    transitions_data.append(transition_data[:, :, non_zero_frames_mask])
        return transitions_data

    def plot_frame(self, transition_index, frame_index):
        if self.visualize and transition_index < len(self.transitions_data):
            self.ax.clear()
            transition_data = self.transitions_data[transition_index]
            frame_data = transition_data[:, :, frame_index]
            self.ax.set_title(f"Transition {transition_index + 1}/{len(self.transitions_data)}, Frame: {frame_index + 1}/{transition_data.shape[-1]}")
            
            for x, y, z in frame_data.T:
                self.ax.scatter(x, y, z, c='r', marker='o')
            
            for start_bone, end_bone in HAND_BONES_CONNECTIONS:
                start_idx = HAND_BONES.index(start_bone)
                end_idx = HAND_BONES.index(end_bone)
                self.ax.plot([frame_data[0, start_idx], frame_data[0, end_idx]],
                             [frame_data[1, start_idx], frame_data[1, end_idx]],
                             [frame_data[2, start_idx], frame_data[2, end_idx]], 'r')
            plt.draw()

    def on_key(self, event):
        if self.visualize:
            if event.key == 'right':
                self.current_frame += 1
                if self.current_frame >= self.transitions_data[self.current_transition].shape[-1]:
                    self.current_frame = 0
                    self.current_transition += 1
                    if self.current_transition >= len(self.transitions_data):
                        self.current_transition = 0  # Loop back to the first transition
            elif event.key == 'left':
                self.current_frame -= 1
                if self.current_frame < 0:
                    self.current_transition -= 1
                    if self.current_transition < 0:
                        self.current_transition = len(self.transitions_data) - 1  # Loop to the last transition
                    self.current_frame = self.transitions_data[self.current_transition].shape[-1] - 1

            self.plot_frame(self.current_transition, self.current_frame)

    def visualize_or_return_data(self):
        if self.visualize:
            if self.transitions_data:
                self.plot_frame(0, 0)  # Start with the first frame of the first transition
                plt.show()
            else:
                print("No valid transitions found for the given word.")
        else:
            # Return transitions data if not visualizing
            return self.transitions_data

# Example usage
npz_path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_reverse_inc.npz'
word = "ARARAT"
visualize = True  # Set to False to return data instead of visualizing

visualizer = HandTransitionVisualizer(npz_path, word, visualize=visualize)
if visualize:
    visualizer.visualize_or_return_data()
else:
    transitions_data = visualizer.visualize_or_return_data()
    # Handle transitions_data as needed, e.g., print or process further
    print(transitions_data)
