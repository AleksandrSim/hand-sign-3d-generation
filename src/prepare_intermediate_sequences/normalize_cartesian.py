import os
import numpy as np

class Normalization:
    def __init__(self, npz_path):
        self.data = np.load(npz_path)['data']
        print("Data shape:", self.data.shape)

    def print_first_frame_first_letter(self, start_letter):
        for i in range(self.data.shape[1]):  # Iterate over letters
            first_frame = self.data[start_letter, i, :, :, 0]  # Get the first frame data
            if not np.all(first_frame == 0):
                print(f"Coordinates for the sequence {start_letter+1}, letter {i+1} in the first frame:")
                print(first_frame)

    def print_last_significant_frame(self, letter_index):
        # Iterate over the frames in reverse to find the last significant frame
        for frame_index in range(self.data.shape[4] - 1, -1, -1):  # Reverse iteration over frames
            frame_data = self.data[letter_index, :, :, :, frame_index]
            if not np.all(frame_data == 0):
                print(f"Last significant frame for letter {letter_index+1} is frame {frame_index+1}:")
                print(frame_data)
                break  # Exit after finding the last significant frame

if __name__ == '__main__':
    npz_path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/eng_test.npz'
    norm = Normalization(npz_path)
    norm.print_first_frame_first_letter(0)  # Start from the first sequence
    norm.print_last_significant_frame(0)  # Check for the last significant frame in the first sequence
