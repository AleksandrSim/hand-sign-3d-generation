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
        transitions = []
        num_chars = self.data.shape[0]  # Assuming there are 27 characters


        for start_char in range(num_chars):
            for end_char in range(num_chars):  # Include all possible end_chars
                first_transition = self.data[start_char, end_char, :, :, :]

                # Find the last non-zero frame for the first transition
                non_zero_frames_first = np.any(first_transition != 0, axis=(0, 1))  # Check across joints and coordinates

                last_frame_index_first = np.max(np.where(non_zero_frames_first)[0]) if np.any(non_zero_frames_first) else -1

                if last_frame_index_first == -1:
                    continue  # Skip if no significant frames are found

                end_frame_first_transition = first_transition[:, :,  last_frame_index_first:last_frame_index_first+1]

                for next_char in range(num_chars):
                    second_transition = self.data[end_char, next_char, :, :, :]

                    # Find the first significant frame in the second transition
                    non_zero_frames_second = np.any(second_transition != 0, axis=(0, 1))
                    first_frame_index_second = np.min(np.where(non_zero_frames_second)[0]) if np.any(non_zero_frames_second) else -1

                    if first_frame_index_second == -1:
                        continue  # Skip if no significant frames are found

                    start_frame_second_transition = second_transition[:, :, first_frame_index_second:first_frame_index_second+1]

                    # Interpolate between the last frame of the first transition and the first frame of the second transition
                    interpolated_frames = self.linear_interpolate(end_frame_first_transition, start_frame_second_transition)
                    interpolated_frames = np.squeeze(interpolated_frames, axis=-1)  # This removes the last dimension if it's 1
        

                    combined_transition = np.concatenate(
                        [first_transition[:, :, :last_frame_index_first],
                        interpolated_frames],
                        axis=2)  # Concatenating along the frame dimension
                    transitions.append(combined_transition)
                    transition_np =np.array(transitions)
                    print(transition_np.shape)


        return transition_np

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
