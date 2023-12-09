import argparse
import json
import os
import sys
# Related third-party imports
import tkinter as tk
from tkinter import ttk
import webbrowser

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from src.process_data.process_data import ProcessBVH

# Local application/library-specific imports
# think how you can efficinetly get rid of this process

current_script_path = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("-i", "--input", type=str,
                        required=True, help="Input file",)
    parser.add_argument("-o", "--output", type=str,
                        required=True, help="Output file or directory path.",)
    args = parser.parse_args()
    return args


class Application(tk.Tk):
    def __init__(self, bvh_reader, json_filepath):
        super().__init__()
        self.bvh_reader = bvh_reader
        self.json_filepath = json_filepath
        self.frame_letter_mapping = self.load_existing_mapping()
        self.title("BVH Visualizer")
        self.geometry("800x600")

        # Mapped character label positioned at the top left
        self.mapped_char_label = ttk.Label(
            self, text="No character mapped for this frame.")
        self.mapped_char_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=10)

        # The rest of the widgets are packed below the mapped_char_label
        self.character_label = ttk.Label(self, text="Enter Character:")
        self.character_label.pack(pady=10)

        self.character_entry = ttk.Entry(self)
        self.character_entry.pack(pady=10)

        self.frame_number_label = ttk.Label(self, text="Enter Frame Number:")
        self.frame_number_label.pack(pady=10)

        self.frame_number_entry = ttk.Entry(self)
        self.frame_number_entry.pack(pady=10)

        self.visualize_button = ttk.Button(
            self,
            text="Visualize with Plotly",
            command=lambda: self.visualize(
                use_plotly=True))
        self.visualize_button.pack(pady=20)

        self.visualize_mpl_button = ttk.Button(
            self,
            text="Visualize with Matplotlib",
            command=lambda: self.visualize(
                use_plotly=False))
        self.visualize_mpl_button.pack(pady=20)

        # Placeholder for the plot
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.bind('<Right>', lambda event: self.change_frame(1))
        self.bind('<Left>', lambda event: self.change_frame(-1))

        # 'n' for next
        self.bind('<n>', lambda event: self.jump_to_marked_frame(1))
        # 'p' for previous
        self.bind('<m>', lambda event: self.jump_to_marked_frame(-1))

    def load_existing_mapping(self):
        """
        Within initiated json_filepath parameter, searches for the JSON object to read.
        Returns the dictionary if object is found, otherwise, empty dictionary.

        Args:
            self
        
        Returns:
            dict: JSON file from json_filepath, otherwise empty dict
        """
        if os.path.exists(self.json_filepath):
            with open(self.json_filepath, 'r') as file:
                return json.load(file)
        return {}

    def change_frame(self, delta):
        """
        Gets the frame we're currently at, and increments it with the parameter argument delta frames
        If valid frame is provided, calls self.visualize function to take the input key of the frame

        Args:
            delta (int): number of incremental frames to jump to
        
        Returns:
            Void: calls visualize function
        """
        try:
            current_frame = int(self.frame_number_entry.get())
            # Ensures frame number doesn't go below 0
            new_frame = max(0, current_frame + delta)
            self.frame_number_entry.delete(0, tk.END)
            self.frame_number_entry.insert(0, str(new_frame))
            # or False, depending on your preference
            self.visualize(use_plotly=False)
        except ValueError:
            print("Current frame number is invalid.")

    def visualize(self, use_plotly=False):
        """
        Takes character from the user input field for the given frame
        Checks if the character is valid persists the frame to character mapping

        Args:
            use_plotly (bool): used for the visualize_joint_locations function call input
        
        Returns:
            Void: persists frame to char mapping
        """
        try:
            ## Given the .visualize function is used internally only through the change_frame function
            ## we might as well take the frame_to_visualize, rather than call is independently (thoughts???)
            frame_to_visualize = int(self.frame_number_entry.get())
            existing_char = self.frame_letter_mapping.get(
                str(frame_to_visualize))

            # Get the character from the entry field
            character = self.character_entry.get().strip().upper()

            if character:
                # Check if the character exists in the
                # latin_to_cyrillic_mapping
                if character in latin_to_cyrillic_mapping:
                    # If a new character is entered, update the mapping and
                    # label
                    self.frame_letter_mapping[str(
                        frame_to_visualize)] = character
                    self.save_mapping()
                    self.mapped_char_label.config(
                        text=f"Frame {frame_to_visualize} is mapped to '{character}'")

                else:
                    # If the character is not in the mapping
                    self.mapped_char_label.config(
                        text=f"The character '{character}' does not exist in the mapping.")
            elif existing_char:
                # If there is an existing character for the frame, update the
                # label
                self.mapped_char_label.config(
                    text=f"Frame {frame_to_visualize} is mapped to '{existing_char}'")
            else:
                # If there is no character for the frame, update the label
                self.mapped_char_label.config(
                    text="No character mapped for this frame.")

            # Clear the character entry field
            self.character_entry.delete(0, tk.END)

            # Visualization with Plotly or Matplotlib
            if use_plotly:
                # Plotly visualization (opens in a web browser)
                html_file = self.bvh_reader.visualize_joint_locations(
                    frame_to_visualize, use_plotly=True)
                webbrowser.open('file://' + html_file)
            else:
                # Matplotlib visualization (embedded in the GUI)
                fig = self.bvh_reader.visualize_joint_locations(
                    frame_to_visualize, use_plotly=False)
                self.show_plot_in_gui(fig)

        except ValueError:
            self.mapped_char_label.config(
                text="Please enter a valid integer for the frame number.")

    def jump_to_marked_frame(self, direction):
        """
        Given the direction of the marked frames we want to move determines and 
        returns the delta of the frames to jump

        Args:
            direction (int): the direction, negative or pos to jump to the marked frame
        
        Returns:
            Void: calls change_frame function with respective delta to jump to the next marked frame

        """
        try:
            current_frame = int(self.frame_number_entry.get())
            marked_frames = sorted(
                [int(frame) for frame in self.frame_letter_mapping.keys()])

            if direction > 0:  # Next marked frame
                next_frames = [
                    frame for frame in marked_frames if frame > current_frame]
                if next_frames:
                    self.change_frame(next_frames[0] - current_frame)
            else:  # Previous marked frame
                prev_frames = [
                    frame for frame in marked_frames if frame < current_frame]
                if prev_frames:
                    self.change_frame(prev_frames[-1] - current_frame)
        except ValueError:
            print("Current frame number is invalid.")

    def show_plot_in_gui(self, fig):
        """
        Clears the previous plot and embeds and displays a new Matplotlib plot within a Tkinter frame

        Args:
            fig: the previous fig object we will clear
        
        Returns:
            void: draws the new Matplotlib plot within a Tkinter frame
        """
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Embed Matplotlib plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def save_mapping(self):
        """
        Helper function for updating (writing) the JSON file with the new mapping 
        
        """
        with open(self.json_filepath, 'w') as file:
            json.dump(self.frame_letter_mapping, file, indent=4)


if __name__ == '__main__':
    args = parse_arguments()
#    BVH_PATH = "/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/datasets/BVH/3D_alphabet_11_15_2023_BVH.bvh"  # Replace with the path to your BVH file
#    JSON_PATH = '/Users/aleksandrsimonyan/Desktop/annotations.json'
    BVH_PATH = args.input
    JSON_PATH = args.output
    bvh_reader = ProcessBVH(BVH_PATH)
    app = Application(bvh_reader, JSON_PATH)
    app.mainloop()
