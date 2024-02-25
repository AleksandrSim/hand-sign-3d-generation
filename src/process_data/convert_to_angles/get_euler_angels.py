import os


import numpy as np
from src.process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS


class JointDataConverter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.joint_data = None
        self.euler_angles = {}
        self.load_data()
    
    def load_data(self):
        # Load the joint data from the .npz file
        self.joint_data = np.load(self.file_path)['data']
    
    def calculate_euler_angles(self, parent_pos, joint_pos, child_pos):
        # Calculate vectors
        bone_vector = joint_pos - parent_pos
        next_bone_vector = child_pos - joint_pos
        
        bone_vector_norm = bone_vector / np.linalg.norm(bone_vector, axis=2, keepdims=True)
        next_bone_vector_norm = next_bone_vector / np.linalg.norm(next_bone_vector, axis=2, keepdims=True)
        
        bone_plane_normal = np.cross(bone_vector_norm, next_bone_vector_norm, axis=2)
        bone_plane_normal_norm = bone_plane_normal / np.linalg.norm(bone_plane_normal, axis=2, keepdims=True)
        
        # Yaw is the angle between the bone_vector projection on the XY plane and the global X axis
        yaw = np.arctan2(bone_vector_norm[:, :, 1, :], bone_vector_norm[:, :, 0, :])
        
        # Pitch is the angle between the bone_vector and its projection on the XY plane
        pitch = np.arctan2(bone_vector_norm[:, :, 2, :], np.sqrt(bone_vector_norm[:, :, 0, :]**2 + bone_vector_norm[:, :, 1, :]**2))
        
        # Simplified roll calculation using the Z component directly
        cos_angle = bone_plane_normal_norm[:, :, 2, :]  # Directly use the Z component for cos_angle
        roll = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Combine the angles into a single array
        euler_angles = np.stack((pitch, roll, yaw), axis=2)
        
        return euler_angles
    
    def convert_to_euler(self):
        # Assume HAND_BONES is defined with the correct order of bone names
        self.euler_angles = {bone: np.zeros((32, 32, 3, 1050)) for bone in HAND_BONES}
        
        for i, bone in enumerate(HAND_BONES):
            if i == 0 or i == len(HAND_BONES) - 1:
                # Skip the first and last bone since they don't have both a parent and a child
                continue
            
            parent_name = HAND_BONES[i-1]
            child_name = HAND_BONES[i+1]
            
            parent_pos = self.joint_data[:, :, :, i-1, :]
            joint_pos = self.joint_data[:, :, :, i, :]
            child_pos = self.joint_data[:, :, :, i+1, :]
            
            # Calculate Euler angles and store them
            self.euler_angles[bone] = self.calculate_euler_angles(parent_pos, joint_pos, child_pos)
    
    def save_data(self, save_path):
        # Save the euler_angles dictionary to a .npz file
        np.savez_compressed(save_path, **self.euler_angles)


if __name__ == "__main__":
    file_path = '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_reverse_inc.npz'
    converter = JointDataConverter(file_path)
    converter.convert_to_euler()
    base_name = os.path.basename(file_path)
    name_without_extension = os.path.splitext(base_name)[0]
    save_dir = os.path.dirname(file_path)
    save_path = os.path.join(save_dir, f"{name_without_extension}_euler_angles.npz")
    print(converter.euler_angles.keys())
    print(converter.euler_angles['RightFinger2Distal'].shape)
    exit()

    converter.save_data(save_path)
    print(f"Euler angles saved to {save_path}")