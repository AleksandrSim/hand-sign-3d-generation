import os
import json

import numpy as np

from src.process_data.utils import char_index_map
print(char_index_map)

class UnifiedDataBuilder:
    def __init__(self, json_path, npz_dir, output_path):
        self.json_path = json_path
        self.npz_dir = npz_dir
        self.output_path = output_path

    def load_json(self):
        with open(self.json_path, 'r') as file:
            return json.load(file)

    def load_npz(self, npz_file):
        return np.load(os.path.join(self.npz_dir, npz_file.replace('json', 'npz')))

    def find_max_transition_length(self, json_data):
        return max(int(duration) for _, (duration, _, _) in json_data.items())

    def extract_and_compile_data(self):
        json_data = self.load_json()
        max_transition_length = self.find_max_transition_length(json_data)
        compiled_data = np.zeros((len(char_index_map), len(char_index_map), 3, 20, max_transition_length))
        skipped_transitions = []

        for transition, (duration, start_frame, npz_file) in json_data.items():

            print(transition)
            start_letter, end_letter = transition.split('_')
            start_frame = int(start_frame)
            duration = int(duration)
            npz_data = self.load_npz(npz_file)['data']
            # Adjust to reflect the correct starting point and duration
            if start_frame + duration - 1 <= npz_data.shape[2]:
                extracted_keypoints = npz_data[:, :, start_frame-1:start_frame-1+duration]
                print(extracted_keypoints)
            else:
                skipped_transitions.append(transition)
                continue  # Skip this iteration if the range is out of bounds
            start_index = char_index_map[start_letter]
            end_index = char_index_map[end_letter]

            for kp_index in range(extracted_keypoints.shape[0]):
                    for xyz_index in range(3):
                        compiled_data[start_index, end_index, xyz_index, kp_index, :duration] = extracted_keypoints[kp_index, xyz_index, :]

            reverse_transition = end_letter + '_' + start_letter
            if reverse_transition not in json_data:
                for kp_index in range(extracted_keypoints.shape[0]):
                    for xyz_index in range(3):
                        # Find the actual length of meaningful data to avoid reversing the padding
                        actual_data_length = np.max(np.where(extracted_keypoints[kp_index, xyz_index, :] != 0)) + 1
                        # Reverse the meaningful data
                        reversed_data = extracted_keypoints[kp_index, xyz_index, :actual_data_length][::-1]
                        # Insert the reversed data into the array, preserving padding at the end
                        compiled_data[end_index, start_index, xyz_index, kp_index, :actual_data_length] = reversed_data


        np.savez_compressed(self.output_path, data=compiled_data)
        if skipped_transitions:
            print("Skipped transitions due to mismatched duration or out-of-bounds frame indices:", skipped_transitions)


if __name__ == "__main__":
    json_path = "/Users/aleksandrsimonyan/Desktop/complete_sequence/english_full/adjust_all_eng.json"
    npz_dir = "/Users/aleksandrsimonyan/Desktop/complete_sequence/english_full/npz"
    output_path = "/Users/aleksandrsimonyan/Desktop/complete_sequence/english_full/master_eng.npz"
    builder = UnifiedDataBuilder(json_path, npz_dir, output_path)
    builder.extract_and_compile_data()
