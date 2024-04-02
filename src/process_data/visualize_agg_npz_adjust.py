import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk

# Assuming the following modules are in your working directory or Python path
from src.process_data.utils import char_index_map
from utils import HAND_BONES, HAND_BONES_CONNECTIONS

class HandTransitionVisualizer:
    def __init__(self, data, start_char, end_char):
        self.data = data
        self.start_char = start_char
        self.end_char = end_char
        self.start_index = char_index_map[start_char]
        self.end_index = char_index_map[end_char]
        self.current_frame = 0
        self.transition_data = self.data[self.start_index, self.end_index, :, :, :]
        self.coordinate_entries = []
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Hand Transition Visualizer")

        # Adjust the figure size to be smaller to make space for controls
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_panel = tk.Frame(self.root)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        for idx, bone in enumerate(HAND_BONES):
            row_frame = tk.Frame(self.control_panel)
            row_frame.pack(fill=tk.X, pady=2)
            tk.Label(row_frame, text=f"{bone}:").pack(side=tk.LEFT)

            x_entry = tk.Entry(row_frame, width=5)
            y_entry = tk.Entry(row_frame, width=5)
            z_entry = tk.Entry(row_frame, width=5)
            x_entry.pack(side=tk.LEFT)
            y_entry.pack(side=tk.LEFT)
            z_entry.pack(side=tk.LEFT)

            self.coordinate_entries.append((x_entry, y_entry, z_entry))

        nav_frame = tk.Frame(self.control_panel)
        nav_frame.pack(fill=tk.X, pady=5)
        tk.Button(nav_frame, text="<< Prev", command=self.prev_frame).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next >>", command=self.next_frame).pack(side=tk.LEFT)

        tk.Button(self.control_panel, text="Update", command=self.update_coordinates).pack(pady=5)

        self.plot_frame(self.current_frame)

    def plot_frame(self, frame):
        self.ax.clear()
        self.ax.set_title(f"Frame: {frame + 1}/{self.transition_data.shape[2]}")
        keypoints = self.transition_data[:, :, frame]

        for idx, (x, y, z) in enumerate(keypoints.T):
            self.ax.scatter(x, y, z, color='red', marker='o')
            self.ax.text(x, y, z, f'{HAND_BONES[idx]}', color='blue')
            x_entry, y_entry, z_entry = self.coordinate_entries[idx]
            x_entry.delete(0, tk.END)
            x_entry.insert(0, f"{x:.2f}")
            y_entry.delete(0, tk.END)
            y_entry.insert(0, f"{y:.2f}")
            z_entry.delete(0, tk.END)
            z_entry.insert(0, f"{z:.2f}")

        for start_bone, end_bone in HAND_BONES_CONNECTIONS:
            start_idx = HAND_BONES.index(start_bone)
            end_idx = HAND_BONES.index(end_bone)
            self.ax.plot([keypoints[0, start_idx], keypoints[0, end_idx]],
                         [keypoints[1, start_idx], keypoints[1, end_idx]],
                         [keypoints[2, start_idx], keypoints[2, end_idx]], color='blue')

        self.canvas.draw()

    def update_coordinates(self):
        frame_data = self.transition_data[:, :, self.current_frame]
        for idx, (x_entry, y_entry, z_entry) in enumerate(self.coordinate_entries):
            try:
                x, y, z = float(x_entry.get()), float(y_entry.get()), float(z_entry.get())
                frame_data[:, idx] = [x, y, z]
            except ValueError:
                continue  # Skip invalid entries
        self.transition_data[:, :, self.current_frame] = frame_data
        self.plot_frame(self.current_frame)

    def next_frame(self):
        if self.current_frame < self.transition_data.shape[2] - 1:
            self.current_frame += 1
            self.plot_frame(self.current_frame)

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.plot_frame(self.current_frame)

    def visualize(self):
        self.root.mainloop()

if __name__ == '__main__':
    path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/eng/test.npz'
    data = np.load(path)['data']
    start_char = 'B'
    end_char = 'B'
    visualizer = HandTransitionVisualizer(data, start_char, end_char)
    visualizer.visualize()
