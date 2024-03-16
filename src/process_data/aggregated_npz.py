import os
import json

import numpy as np

char_index_map = {
    'A': 0, 'B': 1, 'CH': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'HARD': 8, 'I': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'R': 16, 'S': 17, 'SH': 18,
    'SHCH': 19, 'SOFT': 20, 'T': 21, 'TS': 22, 'U': 23, 'V': 24, 'Y': 25, 'YA': 26,
    'YI': 27, 'YO': 28, 'YU': 29, 'Z': 30, 'ZH': 31, 'EE':32, 'space': 33
}


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
        compiled_data = np.zeros((len(char_index_map), len(char_index_map), 3, 19, max_transition_length))
        skipped_transitions = []

        for transition, (duration, start_frame, npz_file) in json_data.items():
            start_letter, end_letter = transition.split('_')
            start_frame = int(start_frame)
            duration = int(duration)
            npz_data = self.load_npz(npz_file)['data']
            
            # Adjust to reflect the correct starting point and duration
            if start_frame + duration - 1 <= npz_data.shape[2]:
                extracted_keypoints = npz_data[:, :, start_frame-1:start_frame-1+duration]
            else:
                skipped_transitions.append(transition)
                continue  # Skip this iteration if the range is out of bounds

            start_index = char_index_map[start_letter]
            end_index = char_index_map[end_letter]

            for kp_index in range(extracted_keypoints.shape[0]):
                for xyz_index in range(3):
                    compiled_data[start_index, end_index, xyz_index, kp_index, :duration] = extracted_keypoints[kp_index, xyz_index, :]

        np.savez_compressed(self.output_path, data=compiled_data)
        if skipped_transitions:
            print("Skipped transitions due to mismatched duration or out-of-bounds frame indices:", skipped_transitions)


if __name__ == "__main__":
    json_path = "/Users/aleksandrsimonyan/Desktop/complete_sequence/output.json"
    npz_dir = "/Users/aleksandrsimonyan/Desktop/complete_sequence/all_npz"
    output_path = "/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data.npz"
    builder = UnifiedDataBuilder(json_path, npz_dir, output_path)
    builder.extract_and_compile_data()
