import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from src.models.hand_sequence import GestureSmoothingLSTM
from src.process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS, letter_to_index, coordinates_input_gt


class GestureVisualizer:
    def __init__(self, predicted_frames):
        self.predicted_frames = predicted_frames
        self.current_frame_idx = 0
        self.max_frame_idx = predicted_frames.shape[0] - 1
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})
        plt.subplots_adjust(bottom=0.2)

        self.axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(self.axnext, 'Next')
        self.bprev = Button(self.axprev, 'Previous')
        self.bnext.on_clicked(self.next_frame)
        self.bprev.on_clicked(self.prev_frame)

    def plot_frame(self, frame_idx):
        frame_data = self.predicted_frames[frame_idx].detach().numpy()
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
        self.current_frame_idx = (self.current_frame_idx + 1) % (self.max_frame_idx + 1)
        self.plot_frame(self.current_frame_idx)

    def prev_frame(self, event):
        self.current_frame_idx = (self.current_frame_idx - 1) % (self.max_frame_idx + 1)
        self.plot_frame(self.current_frame_idx)

    def show(self):
        self.plot_frame(self.current_frame_idx)
        plt.show()

if __name__ == '__main__':

    num_letters = len(letter_to_index)

    letter_embedding_dim = 10  
    hidden_size = 128  
    num_layers = 2  
    keypoints = 19  
    coords = 3  
    sequence_length = 111  


    model = GestureSmoothingLSTM(num_letters, letter_embedding_dim, hidden_size,
                                  num_layers, keypoints, coords, sequence_length)
    checkpoint_path = '/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/model_weights/lstm_fixed_start_epoch_109.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()


    # Prepare test data
    start_letter = 'A'  # Example start letter
    end_letter = 'B'   # Example end letter
    start_letter_idx = torch.tensor([letter_to_index[start_letter]], dtype=torch.long)
    end_letter_idx = torch.tensor([letter_to_index[end_letter]], dtype=torch.long)
    start_coords = torch.tensor(coordinates_input_gt[start_letter].reshape(-1), dtype=torch.float32).unsqueeze(0)
    end_coords = torch.tensor(coordinates_input_gt[end_letter].reshape(-1), dtype=torch.float32).unsqueeze(0)

    # Predict the intermediate frames
    with torch.no_grad():
        predicted_frames = model(start_letter_idx, end_letter_idx, start_coords, end_coords)

    print(predicted_frames.shape)
    exit()
    # Visualize the results
    visualizer = GestureVisualizer(predicted_frames)
    visualizer.show()
