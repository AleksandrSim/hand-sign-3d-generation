import os
import numpy as np

class Normalization:
    def __init__(self, npz_path):
        self.data = np.load(npz_path)['data']
        print("Data shape:", self.data.shape)

    def print_first_frame_first_letter(self, start_letter):
        for i in range(self.data.shape[1]):
            first_frame = self.data[start_letter, i, :, :, 0]
            if not np.all(first_frame == 0):
                print(f"Coordinates for the sequence {start_letter+1}, letter {i+1} in the first frame:")
                print(first_frame)


    def print_dynamic_start_last_frame(self, fixed_end_letter):
        for start_letter in range(self.data.shape[0]):
            non_zero_frames = np.any(self.data[start_letter, fixed_end_letter, :, :, :], axis=(1, 2))
            if np.any(non_zero_frames):
                last_non_zero_frame_index = np.max(np.where(non_zero_frames)[0])
                last_frame = self.data[start_letter, fixed_end_letter, :, :, last_non_zero_frame_index]
                print(f"Coordinates for the sequence {start_letter+1}, letter {fixed_end_letter+1} in the last significant frame:")
                print(last_frame)
            else:
                print(f"No significant frames found for sequence {start_letter+1}, letter {fixed_end_letter+1}.")


if __name__ == '__main__':
    npz_path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/eng_test.npz'
    norm = Normalization(npz_path)
    norm.print_first_frame_first_letter(0)  # Assuming you want to start from the first sequence
    norm.print_dynamic_start_last_frame(0)  # Assuming the fixed end letter is 27th
