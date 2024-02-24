import os
import sys
import argparse

import bpy
import numpy as np

sys.path.append('')
#TODO get rid of sys!

from src.process_data.utils import HAND_BONES


def ArgumentParser():
    parser =  argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help = 'Path to the input')
    parser.add_argument('-o', '--output', help = 'Path to the output')
    return parser.parse_args()


class NumpyConverter:
    def __init__(self, input_path: str, output_path = None):
        self.input_path = input_path
        self.output_path = output_path

    def get_max_frame(self, armature):
        action = armature.animation_data.action
        max_frame_end = action.frame_range[1]
        return max_frame_end

    def iterate_folder(self):
        fbx_files = os.listdir(self.input_path)
        fbx_files = [i for i in fbx_files if i.endswith('.fbx')]

        for fbx_file in fbx_files:
            print(f'fbx_file {fbx_file}')
            fbx = self.open_fbx(os.path.join(self.input_path, fbx_file))
            max_frame_number = int(self.get_max_frame(fbx))
            print(f'max_frame_number {max_frame_number }')

            np_array = self.get_coordinates(fbx, max_frame_number)
            self.save_npz(np_array, fbx_file)
            print(f'save_npz {fbx_file}')

    def open_fbx(self, fbx_path):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE':
                return obj 
            
    def get_coordinates(self, fbx, max_frame_number, joint_names = None):
        # Get joint names
        # TODO fix the joint_names as global var to avoid misordering

        if not joint_names:
            joint_names = [bone.name for bone in fbx.pose.bones]

        length = len(HAND_BONES)
        np_array = np.zeros((length, 3, int(max_frame_number)))
        for frame in range(max_frame_number):
            bpy.context.scene.frame_set(frame)
            # Iterate over each joint
            for idx, joint in enumerate(HAND_BONES):
                print(len(HAND_BONES))

                print(f'frame_number {frame}')
                assert joint in joint_names 

                coordinates = fbx.pose.bones[joint].head
                np_array[idx, :, frame] = [coordinates.x, coordinates.y, coordinates.z]
        return np_array

    def save_npz(self, np_array, fbx_file):
        np.savez(os.path.join(self.output_path, fbx_file.replace('fbx', 'npz')), data=np_array)


if __name__ == '__main__':
    args = ArgumentParser()
    inp, out = args.input, args.output
    converter = NumpyConverter(input_path=inp, output_path=out)
    converter.iterate_folder()


        
            
    

