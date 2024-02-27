import numpy as np
from src.process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS


class HandJointAngleCalculator:
    def __init__(self, data):
        """
        Initialize the calculator with the 3D keypoints data.
        :param data: A NumPy array of shape (32, 32, 3, 19, 1050).
        """
        self.data = data

    def calculate_angle_between_vectors(self, vector1, vector2):
        """
        Calculate the angle between two vectors for each frame.
        """
        dot_product = np.sum(vector1 * vector2, axis=-1)
        norm1 = np.linalg.norm(vector1, axis=-1)
        norm2 = np.linalg.norm(vector2, axis=-1)
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure it's within the valid range for arccos
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)

    def calculate_joint_angles(self):
        """
        Calculate angles at each joint for all frames, iterating over frames.
        """
        angles_dict = {}
        num_frames = self.data.shape[-1]

        for i, (start_bone, end_bone) in enumerate(HAND_BONES_CONNECTIONS[:-1]):
            if HAND_BONES_CONNECTIONS[i+1][0] != end_bone:
                continue

            next_bone = HAND_BONES_CONNECTIONS[i+1][1]

            # Indices in the data for each point
            idx1, idx2, idx3 = HAND_BONES.index(start_bone), HAND_BONES.index(end_bone), HAND_BONES.index(next_bone)

            # Initialize an empty array to store angles for all frames
            all_frames_angles = np.zeros((32, 32, 3, num_frames))

            for frame in range(num_frames):
                # Extracting points for vector calculation for the current frame
                p_start = self.data[:, :, :, idx1, frame]
                p_joint = self.data[:, :, :, idx2, frame]
                p_end = self.data[:, :, :, idx3, frame]

                # Calculating vectors AB and BC for the current frame
                vector_AB = p_joint - p_start
                vector_BC = p_end - p_joint

                # Calculating the angle at joint B for the current frame
                angle = self.calculate_angle_between_vectors(vector_AB, vector_BC)

                # Store the calculated angle for the current frame
                all_frames_angles[:, :, :, frame] = angle

            key = f'{start_bone}-{end_bone}-{next_bone}'
            angles_dict[key] = all_frames_angles

        return angles_dict

if __name__ == '__main__':
    path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_reverse_inc.npz'
    output = '/Users/aleksandrsimonyan/Desktop/complete_sequence/joint_angles.npz'
    data = np.load(path)['data']
    print(data.shape)
    calculator = HandJointAngleCalculator(data)
    joint_angles = calculator.calculate_joint_angles()

    # Checking the shape of angles for a specific joint
    example_key = 'RightFinger5Proximal-RightFinger5Medial-RightFinger5Distal'
    print(joint_angles[example_key].shape)  # Expected to print (32, 32, 3, 1050)

    calculator.save_data(joint_angles, output)