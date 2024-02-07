import os
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aggregated_npz import char_index_map

from utils import HAND_BONES, HAND_BONES_CONNECTIONS
HAND_BONES_INDEXES = list(range(19))


char_index_map = {
    'A': 0, 'B': 1, 'CH': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'HARD': 8, 'I': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'R': 16, 'S': 17, 'SH': 18,
    'SHCH': 19, 'SOFT': 20, 'T': 21, 'TS': 22, 'U': 23, 'V': 24, 'Y': 25, 'YA': 26,
    'YI': 27, 'YO': 28, 'YU': 29, 'Z': 30, 'ZH': 31
}


class HandTransitionVisualizer:
    def __init__(self, data, start_char, end_char):
        self.data = data
        self.start_char = start_char
        self.end_char = end_char
        self.start_index = char_index_map[start_char]
        self.end_index = char_index_map[end_char]
        self.current_frame = 0
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.gcf().canvas.mpl_connect('key_press_event', self.on_key)
        self.transition_data  = self.data[self.start_index, self.end_index, :,:,:]
        print(self.transition_data.shape)
        non_zero_frames_mask = np.any(self.transition_data != 0, axis=(0, 1))
        self.transition_data = self.transition_data[:, :,  non_zero_frames_mask]


    def plot_frame(self, frame):
        self.ax.clear()  # Clear existing points and lines
        self.ax.set_title(f"Frame: {frame + 1}/{self.transition_data.shape[-1]}")  # Update title with current frame

        keypoints = self.transition_data[:, :, frame]

        for x, y, z in keypoints.T:
            self.ax.scatter(x, y, z, c='r', marker='o')

        for start_bone, end_bone in HAND_BONES_CONNECTIONS:
            start_idx = HAND_BONES.index(start_bone)
            end_idx = HAND_BONES.index(end_bone)
            self.ax.plot([keypoints[0, start_idx], keypoints[0, end_idx]],
                         [keypoints[1, start_idx], keypoints[1, end_idx]],
                         [keypoints[2, start_idx], keypoints[2, end_idx]], 'r')

        plt.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_frame = min(self.current_frame + 1, self.transition_data.shape[-1] - 1)
        elif event.key == 'left':
            self.current_frame = max(self.current_frame - 1, 0)
        self.plot_frame(self.current_frame)

    def visualize(self):
        self.plot_frame(self.current_frame)
        plt.show()

    def find_missing_transitions(self):
        missing_transitions = []
        characs = char_index_map.keys()

        for start_char in characs:
            for end_char in characs:
                start_index, end_index = char_index_map[start_char], char_index_map[end_char]
                transition_data = self.data[start_index, end_index, :, :, :]
                if not np.any(transition_data):
                    missing_transitions.append((start_char, end_char))

        print(f"Missing Transitions: {len(missing_transitions)} / {len(char_index_map)**2}")
        print(missing_transitions)
        return missing_transitions

    def fill_missing_with_reverse(self):
        missing_transitions = self.find_missing_transitions()
        filled = 0
        filled_list = []

        for start_char, end_char in missing_transitions:
            start_index, end_index = char_index_map[start_char], char_index_map[end_char]
            # Check if the reverse transition exists
            reverse_data = self.data[end_index, start_index, :, :, :]

            if np.any(reverse_data):
                # Reverse the frames and assign to the missing transition
                self.data[start_index, end_index, :, :, :] = reverse_data[:, :, ::-1]
                filled += 1
                filled_list.append([start_char, end_char])
        print(filled_list)
        

        return filled




    def save_to_npz(self, output_path):
        """Save the updated data array to an NPZ file."""
        np.savez_compressed(output_path, data=self.data)
        print(f"Data saved to {output_path}")

if __name__ == '__main__':
    path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_reverse_inc.npz'
    data = np.load(path)['data']
    visualizer = HandTransitionVisualizer(data, 'A', 'TS')
    visualizer.find_missing_transitions()
    visualizer.fill_missing_with_reverse()
#    visualizer.save_to_npz('/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_reverse_inc.npz')
    visualizer.visualize()