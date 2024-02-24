import numpy as np
import matplotlib.pyplot as plt

from src.process_data.utils import HAND_BONES_CONNECTIONS, HAND_BONES, coordinates_input_gt



def plot_hand_sign(coordinates, title='Hand Sign'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for start_bone, end_bone in HAND_BONES_CONNECTIONS:
        start_idx = HAND_BONES.index(start_bone)
        end_idx = HAND_BONES.index(end_bone)
        ax.plot([coordinates[start_idx, 0], coordinates[end_idx, 0]],
                [coordinates[start_idx, 1], coordinates[end_idx, 1]],
                [coordinates[start_idx, 2], coordinates[end_idx, 2]], 'ro-')

    ax.set_title(title)
    plt.show()

# Coordinates for the letter 'S'
plot_hand_sign(coordinates_input_gt['B'])


