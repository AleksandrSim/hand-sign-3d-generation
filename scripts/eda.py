import numpy as np
import bpy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import argparse
import dash
from scipy.spatial import distance
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Assume you have a dictionary 'distances_by_bone' where keys are bones and values are arrays of Euclidean distances

import sys
sys.path.append("/Users/vachemacbook/Desktop/Immensus/hand-sign-3d-generation/src")
from process_data.process_data import ProcessBVH
from process_data.utils import HAND_BONES, HAND_BONES_CONNECTIONS

def calculate_all_distances(bvh_reader, start_frame, end_frame):
    """
    Calculate Euclidean distances between specified bones for all frames in the given range.

    Args:
        bvh_reader (ProcessBVH): Instance of ProcessBVH.
        start_frame (int): Start frame number.
        end_frame (int): End frame number.

    Returns:
        dict: Dictionary with bones as keys and arrays of Euclidean distances for each frame.
    """
    distances_by_bone = {}
    joint_names = bvh_reader.get_all_joint_names()
    for bone in joint_names:
            if bone not in HAND_BONES:
                continue
            else:
                distances_array = []
                prev_loc = bvh_reader.get_bone_location(bone, 0)
                prev_array = (prev_loc.x, prev_loc.y, prev_loc.z)
                for frame in range(start_frame+1, end_frame):
                    curr_loc = bvh_reader.get_bone_location(bone, frame)
                    curr_array = (curr_loc.x, curr_loc.y, curr_loc.z)
                    distances_array.append(np.sqrt(np.sum(((curr_array[0] - prev_array[0]) ** 2) +
                                                          ((curr_array[1] - prev_array[1]) ** 2) +
                                                          ((curr_array[2] - prev_array[2]) ** 2))))
                    prev_array = curr_array
                    del curr_array
            
                distances_by_bone[bone] = distances_array
    return distances_by_bone

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file",)
    parser.add_argument("-e", "--elevation", required=False, default=10, 
                        type=str, help="elevation value for the plot visualization.")    
    parser.add_argument("-a", "--azimuth", required=False, default=10, 
                        type=str, help="azimuth value for the plot visualization.")      
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    BVH_PATH = args.input
    ELEVATION = args.elevation
    AZIMUTH = args.azimuth
    bvh_reader = ProcessBVH(BVH_PATH, ELEVATION, AZIMUTH)

    # Calculate Euclidean distances for all frames in the specified range
    START_FRAME = 0
    END_FRAME = int(bvh_reader.max_frame_end)  # Adjust this to the desired end frame

    # Create a Dash web application
    distances_by_bone = calculate_all_distances(bvh_reader, START_FRAME, END_FRAME)

    # print(distances_by_bone['RightFinger2Medial'].sum())
    # print(distances_by_bone['RightFinger2Medial'].mean())
    

    app = dash.Dash(__name__)

    # Define layout
    app.layout = html.Div([
        html.H1("Interactive Euclidean Distance Visualization"),
        # Dropdown menu for selecting bones
        dcc.Dropdown(
            id='bone-dropdown',
            options=[{'label': bone, 'value': bone} for bone in distances_by_bone.keys()],
            value=list(distances_by_bone.keys()),  # Default: All bones selected
            multi=True,
            style={'width': '50%'}
        ),
        # Line chart for visualizing Euclidean distances
        dcc.Graph(id='line-chart')
    ])

    # Define callback to update line chart based on dropdown selection
    @app.callback(
        Output('line-chart', 'figure'),
        [Input('bone-dropdown', 'value')]
    )
    def update_line_chart(selected_bones):
        fig = px.line()
        for bone in selected_bones:
            fig.add_scatter(x=list(range(3, END_FRAME + 1)),
                            y=distances_by_bone[bone][3:],
                            mode='lines+markers',
                            name=bone)

        # Update layout
        fig.update_layout(title='Euclidean Distance Variation Across Frames',
                          xaxis_title='Frame Number',
                          yaxis_title='Euclidean Distance',
                          showlegend=True)

        return fig

    app.run_server(debug=True)
