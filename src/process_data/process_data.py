# Third-party imports
import bpy
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Local application/library specific imports
from process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS


class ProcessBVH:
    def __init__(self, bvh_path, elevation=10, azimuth=10):
        self.path = bvh_path
        self.armature_name = None
        self.import_bvh()
        self.find_armature()
        bpy.context.scene.frame_end =10000
        self.set_max_frame_end()
        self.elevation = elevation
        self.azimuth = azimuth

        self.mpl_fig = None  # For storing a reference to the Matplotlib figure
        self.mpl_ax = None  

        print(self.max_frame_end)
    
    def set_max_frame_end(self):
        # Assuming the action is associated with the first armature
        armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
        if armatures:
            armature = armatures[0]
            if armature.animation_data and armature.animation_data.action:
                action = armature.animation_data.action
                self.max_frame_end = action.frame_range[1]
            else:
                self.max_frame_end = bpy.context.scene.frame_end
        else:
            self.max_frame_end = bpy.context.scene.frame_end

    def import_bvh(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

        if self.path.lower().endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=self.path)
        elif self.path.lower().endswith('.bvh'):
            bpy.ops.import_anim.bvh(filepath=self.path)
        else:
            raise ValueError("Unsupported file format")

    def set_frame(self, frame):
        bpy.context.scene.frame_set(frame)

    def find_armature(self):
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE':
                self.armature_name = obj.name
                break

    def get_all_joint_names(self):
        """
        A Method for generating a list of joint_names from the self.armature_name

        Args:
            self
        
        Returns:
            list: list of pose.bones.name by the order they appear in self.armature_name

        """
        joint_names = []
        armature = bpy.data.objects[self.armature_name]
        for bone in armature.pose.bones:
            joint_names.append(bone.name)
        return joint_names

    def get_bone_location(self, bone_name, frame):
        """
        For the given the bone name and the frame returns the coordinates

        Args:
            bone_name: bone name
            frame (int): frame number
        
        Returns:
            location: location object dtypes with respective (x, y, z) coordinates in 3D space
            
        """
        self.set_frame(frame)
        armature = bpy.data.objects[self.armature_name]
        return armature.pose.bones[bone_name].head

    def visualize_joint_locations(
            self,
            frame_to_visualize,
            save_path=None,
            use_plotly=False,
            debug=False):
        """
        A function that iterates through all joints and connections.
        Produces a 3D visualization for a specific frame number

        Args:
            frame_to_visualize (int): the frame number for which to produce visualizations
            save_path (str): default=None, the file directory where the visualization html file is saved
            use_plotly (bool): default=False, parameter for using plotly, otherwise matplotlib is used
            debug (bool): default=False, along with the returned object, produces the visualizations
        
        Returns:
            fig: Visualization container with the 3D coordinates for all HAND_BONES and HAND_BONES_CONNECTIONS

        
        """
        joint_names = self.get_all_joint_names()
        joint_locations = {}
        for joint_name in joint_names:
            if joint_name not in HAND_BONES:
                continue
            location = self.get_bone_location(joint_name, frame_to_visualize)
            joint_locations[joint_name] = location
#            print(
#                f"{joint_name} Location at Frame {frame_to_visualize}: X={location.x}, Y={location.y}, Z={location.z}")

        if use_plotly:
            fig = go.Figure()
            for joint_name, location in joint_locations.items():
                fig.add_trace(go.Scatter3d(
                    x=[location.x], y=[location.y], z=[location.z],
                    mode='markers',
                    marker=dict(size=5),
                    name=joint_name
                ))
            for connection in HAND_BONES_CONNECTIONS:
                if connection[0] in joint_locations and connection[1] in joint_locations:
                    loc0 = joint_locations[connection[0]]
                    loc1 = joint_locations[connection[1]]
                    fig.add_trace(go.Scatter3d(
                        x=[loc0.x, loc1.x], y=[
                            loc0.y, loc1.y], z=[loc0.z, loc1.z],
                        mode='lines', line=dict(color='blue', width=2)
                    ))
            fig.update_layout(scene=dict(aspectmode="data"))
            fig.update_layout(
                title=f'Scatter Plot - Frame {frame_to_visualize}')
            if save_path:
                fig.write_html(save_path + '.html')
            fig.show()
        else:
            if self.mpl_fig is None or self.mpl_ax is None:
                # Create the plot if it doesn't exist
                self.mpl_fig, self.mpl_ax = plt.subplots(subplot_kw={'projection': '3d'})
                created_new_plot = True
            else:
                # Clear existing data for update
                self.mpl_ax.cla()
                created_new_plot = False

            # Update plot with new data
            for location in joint_locations.values():
                self.mpl_ax.scatter(location.x, location.y, location.z, 'o')
            for connection in HAND_BONES_CONNECTIONS:
                if connection[0] in joint_locations and connection[1] in joint_locations:
                    loc0 = joint_locations[connection[0]]
                    loc1 = joint_locations[connection[1]]
                    self.mpl_ax.plot([loc0.x, loc1.x], [loc0.y, loc1.y], [loc0.z, loc1.z], 'blue')
            # Set the view angle only if a new plot is created
            if created_new_plot:
                self.mpl_ax.view_init(elev=int(self.elevation), azim=int(self.azimuth))
            self.mpl_ax.set_title(f'3D Axes Plot - Frame {frame_to_visualize}')
            if debug or created_new_plot:
                plt.imshow(self.mpl_fig)
                plt.show()

            return self.mpl_fig


if __name__ == '__main__':
    # Replace with the path to your BVH file
    BVH_PATH = "/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/datasets/BVH/3D_alphabet_11_15_2023_BVH.bvh"
    bpy.context.scene.frame_end = 2147483647  # Set the end frame to the desired value
    FRAME_TO_VISUALIZE = 3400  # Frame number to visualize
    # Replace with your desired save path and file name without extension
    SAVE_PATH = "/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/process"

    bvh_reader = ProcessBVH(BVH_PATH)

    # Visualize as a 3D scatter plot using Plotly and save as HTML
    bvh_reader.visualize_joint_locations(FRAME_TO_VISUALIZE, SAVE_PATH)

#    bvh_reader.visualize_joint_locations(FRAME_TO_VISUALIZE, SAVE_PATH, plot_type='axes3d')
