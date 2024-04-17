import time
import queue
import warnings
import multiprocessing
import numpy as np
import requests

import tkinter as tk
from requests import Session
from tkinter import messagebox

# Importing necessary modules from your custom package
from src.control.config import ALL_CTRL
from src.control.filters import Filters, interpolate_sequences
from src.control.sequences import PHRASES
from src.process_data.utils import char_index_map
from src.control.control_rig import get_control_rig
from src.process_data.sequence import filter_non_zero
from src.control.language_engine import transform_to_list

# Constants for server communication and process handling
ADDRESS = "http://localhost:30010"
ENDPOINT = "/remote/preset/RCP_Hand/function/Set%20Hand%20Pose"
DATA_PATH = "C:\\Users\\asimonyan\\Desktop\\ue5\\master_eng.npz"
FPS = 100
FRAME_TIME = 1.0 / FPS
SKIP_FRAMES_RATE = 2
TIMEOUT = 0.01
SLEEP_TIME = 0.01

filters = Filters()

def send_data(m_queue: multiprocessing.Queue, url: str):
    curr = 0
    session = Session()  # Create a session object
    while True:
        try:
            if not m_queue.empty():
                start_t = time.perf_counter()
                s = m_queue.get(timeout=TIMEOUT)
                payload = {}
                for ctrl in ALL_CTRL:
                    if ctrl in s:
                        print(f'Currently processing {curr}')
                        curr += 1
                        angle = s[ctrl]
                        payload[ctrl] = {"Roll": angle[0], "Pitch": angle[1], "Yaw": angle[2]}
                    else:
                        payload[ctrl] = {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0}

                session.put(url, json={"Parameters": payload, "generateTransaction": True}, timeout=1.0)
                send_time = time.perf_counter() - start_t
                time.sleep(max(FRAME_TIME - send_time, 0.0))
            else:
                time.sleep(SLEEP_TIME)
        except queue.Empty:
            continue
        except Exception as e:
            warnings.warn(f"Error occurred while sending data: {e}", stacklevel=2)

def get_queued_data(txt: list[str], data_queue: multiprocessing.Queue, data):
    if len(txt) <= 1:
        warnings.warn(f"{txt} is not a valid sequence. We support > 1 letters", stacklevel=2)
        return
    seqs = []
    for i in range(len(txt)-1):
        if txt[i] in PHRASES:
            seqs.append((PHRASES[txt[i]], txt[i]))
        elif txt[i] in char_index_map and txt[i+1] in char_index_map:
            if i < len(txt)-2:
                interpolated_data = interpolate_sequences(
                    data, char_index_map[txt[i]],
                    char_index_map[txt[i+1]], char_index_map[txt[i+2]])
                seqs.append((interpolated_data, txt[i]))
            else:
                seqs.append((filter_non_zero(
                    data[char_index_map[txt[i]], char_index_map[txt[i+1]], :, :, :]), txt[i]))

    if txt[-1] in PHRASES:
        seqs.append((PHRASES[txt[-1]], txt[-1]))
    rigs = []
    for seq in seqs:
        rig_seq = [get_control_rig(seq[0][:, :, i])
                   for i in range(0, seq[0].shape[-1], SKIP_FRAMES_RATE)]
        rig_seq = filters(rig_seq)
        rigs.append(rig_seq)
    filters.reset()
    rigs = [pose for rig in rigs for pose in rig]
    for rig in rigs:
        data_queue.put(rig)

def main():
    data = np.load(DATA_PATH)['data']
    endpoint_url = ADDRESS + ENDPOINT
    data_queue = multiprocessing.Queue()

    sender_process = multiprocessing.Process(target=send_data, args=(data_queue, endpoint_url))
    sender_process.start()

    root = tk.Tk()
    root.title("Gesture Control Input")

    input_field = tk.Entry(root, width=50)
    input_field.pack(padx=20, pady=20)

    def on_submit():
        user_input = input_field.get()
        txt_sequence = transform_to_list(user_input, list(PHRASES.keys()))
        get_queued_data(txt=txt_sequence, data_queue=data_queue, data=data)

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(padx=20, pady=20)

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            sender_process.terminate()
            sender_process.join()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()