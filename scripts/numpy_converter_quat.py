import os
import sys
import argparse
import bpy
import numpy as np

# Assuming the directory structure is already in the Python path or correct sys.path.append() is used
from src.process_data.utils import HAND_BONES

def ArgumentParser():
    parser = argparse.ArgumentParser(description="Extract and save quaternions from FBX files")
    parser.add_argument('-i', '--input', help='Path to the input directory containing FBX files')
    parser.add_argument('-o', '--output', help='Path to the output directory to save NPZ files')
    return parser.parse_args()

class NumpyConverter:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def get_max_frame(self, armature):
        action = armature.animation_data.action
        return int(action.frame_range[1])

    def iterate_folder(self):
        fbx_files = [f for f in os.listdir(self.input_path) if f.endswith('.fbx')]
        for fbx_file in fbx_files:
            print(f'Processing file: {fbx_file}')
            fbx = self.open_fbx(os.path.join(self.input_path, fbx_file))
            max_frame_number = self.get_max_frame(fbx)
            quaternion_array = self.get_quaternions(fbx, max_frame_number)
            self.save_npz(quaternion_array, fbx_file)
            print(f'Saved quaternion data for {fbx_file}')

    def open_fbx(self, fbx_path):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE':
                return obj

    def get_quaternions(self, fbx, max_frame_number):
        quaternion_array = np.zeros((len(HAND_BONES), 4, max_frame_number))  # Quaternion has 4 components: w, x, y, z
        for frame in range(1, max_frame_number + 1):  # Frame counting in Blender starts from 1
            bpy.context.scene.frame_set(frame)
            for idx, joint_name in enumerate(HAND_BONES):
                bone = fbx.pose.bones[joint_name]
                quaternion = bone.rotation_quaternion
                quaternion_array[idx, :, frame - 1] = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
        return quaternion_array

    def save_npz(self, np_array, fbx_file):
        npz_filename = os.path.join(self.output_path, fbx_file.replace('.fbx', '.npz'))
        np.savez_compressed(npz_filename, data=np_array)

if __name__ == '__main__':
    args = ArgumentParser()
    converter = NumpyConverter(input_path=args.input, output_path=args.output)
    converter.iterate_folder()
