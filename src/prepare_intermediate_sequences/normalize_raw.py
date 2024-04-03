import numpy as np


class TransitionInterpolator:
    def __init__(self, input_npz_path, output_npz_path, num_steps=30):
        self.input_npz_path = input_npz_path
        self.output_npz_path = output_npz_path
        self.num_steps = num_steps
        self.data = self.load_data()

    def load_data(self):
        return np.load(self.input_npz_path)['data']

    def linear_interpolate(self, start, end):
        return np.linspace(start, end, self.num_steps, axis=2)
    

    def process_transitions(self):
        num_chars = self.data.shape[0]  # 34 characters
        original_frames = self.data.shape[-1]  # 1050 frames

        # If you add 30 frames to each of the 34*34 transitions, the increase would be much larger
        # But if you're adding 30 frames in total for the dataset, then the final frame count would be 1050 + 30
        total_frames = original_frames + 30

        # Initialize the array to hold the transitions
        transitions_data = np.zeros((num_chars, num_chars, 3, 19, total_frames))

        for start_char in range(num_chars):
            for end_char in range(num_chars):
                first_transition = self.data[start_char, end_char, :, :, :]

                non_zero_frames_first = np.any(first_transition != 0, axis=(0, 1))
                last_frame_index_first = np.max(np.where(non_zero_frames_first)[0]) if np.any(non_zero_frames_first) else -1

                if last_frame_index_first == -1:
                    continue

                end_frame_first_transition = first_transition[:, :, last_frame_index_first:last_frame_index_first+1]
                
                # Assume the interpolation is done just once between each unique pair of transitions
                if end_char < num_chars - 1:
                    second_transition = self.data[start_char, end_char + 1, :, :, :]
                    non_zero_frames_second = np.any(second_transition != 0, axis=(0, 1))
                    first_frame_index_second = np.min(np.where(non_zero_frames_second)[0]) if np.any(non_zero_frames_second) else -1

                    if first_frame_index_second == -1:
                        continue

                    start_frame_second_transition = second_transition[:, :, first_frame_index_second:first_frame_index_second+1]

                    interpolated_frames = self.linear_interpolate(end_frame_first_transition, start_frame_second_transition)
                    interpolated_frames = np.squeeze(interpolated_frames, axis=-1)
                    
                    # Fill in the transitions_data array
                    transition_end = min(last_frame_index_first + 1 + 30, total_frames)  # Ensure not to exceed the total frame count
                    transitions_data[start_char, end_char, :, :, :transition_end] = np.concatenate(
                        [first_transition[:, :, :last_frame_index_first+1],
                        interpolated_frames],
                        axis=2
                    )

        return transitions_data


    def save_transitions(self, transitions):
        np.savez(self.output_npz_path, data=transitions)

    def run(self):
        transitions = self.process_transitions()
        self.save_transitions(transitions)

if __name__ == '__main__':
    input_npz_path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_master.npz'
    output_npz_path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_master_normalized.npz'

    interpolator = TransitionInterpolator(input_npz_path, output_npz_path)
    interpolator.run()
