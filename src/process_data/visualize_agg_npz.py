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

    def plot_frame(self, frame):
        self.ax.clear()  # Clear existing points and lines
        self.ax.set_title(f"Frame: {frame + 1}/{self.data.shape[-1]}")  # Update title with current frame

        keypoints = self.data[self.start_index, self.end_index, :, :, frame]

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
            self.current_frame = min(self.current_frame + 1, self.data.shape[-1] - 1)
        elif event.key == 'left':
            self.current_frame = max(self.current_frame - 1, 0)
        self.plot_frame(self.current_frame)

    def visualize(self):
        self.plot_frame(self.current_frame)
        plt.show()



if __name__ == '__main__':
    path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data.npz'
    data = np.load(path)['data']
    visualizer = HandTransitionVisualizer(data, 'B', 'A')  # Specify the start and end characters
    visualizer.visualize()