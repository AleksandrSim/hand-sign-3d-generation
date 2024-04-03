import time
import queue
import warnings
import threading
import multiprocessing

import numpy as np
import requests

from src.control.config import ALL_CTRL
from src.control.filters import Filters
from scripts.retrive_data import filter_non_zero
from src.control.sequences import PHRASES
from src.process_data.utils import char_index_map
from src.control.control_rig import get_control_rig
from src.control.language_engine import transform_to_list

FPS = 30
FRAME_TIME = 1.0 / FPS
SKIP_FRAMES_RATE = 3
TIMEOUT = 0.1
DATA_PATH = 'data/sequence/unified_data_master.npz'
data = np.load(DATA_PATH)['data']
SLEEP_TIME = 0.1  # Sleep time to check if we have a new input

filters = Filters()

def linear_interpolate(start, end, num_steps):
    # This function will create interpolated frames between start and end frames
    return np.linspace(start, end, num_steps, axis=2)


def interpolate_sequences(data, start_idx, end_idx, next_idx=None):
    transition_data = data[start_idx, end_idx, :, :, :]
    non_zero_frames_mask = np.any(transition_data != 0, axis=(0, 1))
    transition_data = transition_data[:, :, non_zero_frames_mask]    

    if next_idx:
        next_transition_data = data[end_idx, next_idx, :, :, :]

        start_frame = transition_data[:, :, -1:]  
        end_frame = next_transition_data[:, :, :1] 
        interpolated_frames = linear_interpolate(start_frame, end_frame, num_steps=120)
        
        interpolated_frames_squeezed = np.squeeze(interpolated_frames, axis=-1)
        transition_data = np.concatenate((transition_data[:, :, :], interpolated_frames_squeezed), axis=2)
    return transition_data


def send_data(m_queue: multiprocessing.Queue, url: str):
    while True:
        try:
            if not m_queue.empty():
                start_t = time.perf_counter()
                s = m_queue.get(timeout=TIMEOUT)
                payload = {}
                for ctrl in ALL_CTRL:
                    if ctrl in s:
                        angle = s[ctrl]
                        payload[ctrl] = {"Roll": angle[0],
                                         "Pitch": angle[1],
                                         "Yaw": angle[2]}
                    else:
                        # Send zeros for the controls we have no info
                        payload[ctrl] = {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0}

                requests.put(url, json={
                    "Parameters": payload,
                    "generateTransaction": True
                },
                    timeout=1.0)
                send_time = time.perf_counter() - start_t
                time.sleep(max(FRAME_TIME-send_time, 0.0))
            else:
                time.sleep(SLEEP_TIME)
        except queue.Empty:  # Here queue is a mudule, not a class entity
            pass
        except Exception as e:
            warnings.warn(f"Error occurred while sending data: {e}")


def listen_input(data_queue: multiprocessing.Queue):
    while True:
        try:
            user_input = input("Enter a string: ")
            txt_sequence = transform_to_list(user_input, list(PHRASES.keys()))
            get_queued_data(txt=txt_sequence, data_queue=data_queue)
        except KeyboardInterrupt:
            break
        time.sleep(SLEEP_TIME)


def get_queued_data(txt: list[str], data_queue: multiprocessing.Queue):
    if len(txt) <= 1:
        warnings.warn(f"{txt} is not a valid sequence. We support > 1 letters")
        return
    seqs: list[tuple[np.ndarray, str]] = []
    for i in range(len(txt)-1):
        if txt[i] in PHRASES:
            seqs.append((PHRASES[txt[i]], txt[i]))
        elif txt[i] in char_index_map and txt[i+1] in char_index_map:
            seqs.append((filter_non_zero(
                data[char_index_map[txt[i]],
                     char_index_map[txt[i+1]], :, :, :]), txt[i]))
    if txt[-1] in PHRASES:
        # Deal with the case when a keyphrase at the end of the sequence
        seqs.append((PHRASES[txt[i]], txt[-1]))
    rigs = []
    for seq in seqs:
        rig_seq = [get_control_rig(seq[0][:, :, i])
                   for i in range(0, seq[0].shape[-1], SKIP_FRAMES_RATE)]
        rig_seq = filters(rig_seq)
        rigs.append(rig_seq)
    filters.reset()  # Reset filters after processing the phrase
    rigs = [pose for rig in rigs for pose in rig]
    for rig in rigs:
        data_queue.put(rig)

def get_queued_data_normalized(txt: list[str], data_queue: multiprocessing.Queue):
    if len(txt) <= 1:
        warnings.warn(f"{txt} is not a valid sequence. We support > 1 letters")
        return
    seqs: list[tuple[np.ndarray, str]] = []
    for i in range(len(txt)-1):
        if txt[i] in PHRASES:
            seqs.append((PHRASES[txt[i]], txt[i]))
        elif txt[i] in char_index_map and txt[i+1] in char_index_map:
            if i < len(txt)-2:
                interpolated_data = interpolate_sequences(data, 
                                      char_index_map[txt[i]], char_index_map[txt[i+1]], char_index_map[txt[i+2]])
            seqs.append(interpolated_data, txt[i]))
            

    if txt[-1] in PHRASES:
        # Deal with the case when a keyphrase at the end of the sequence
        seqs.append((PHRASES[txt[i]], txt[-1]))
    rigs = []
    for seq in seqs:
        rig_seq = [get_control_rig(seq[0][:, :, i])
                   for i in range(0, seq[0].shape[-1], SKIP_FRAMES_RATE)]
        rig_seq = filters(rig_seq)
        rigs.append(rig_seq)
    filters.reset()  # Reset filters after processing the phrase
    rigs = [pose for rig in rigs for pose in rig]
    for rig in rigs:
        data_queue.put(rig)



    if txt[-1] in PHRASES:
        # Deal with the case when a keyphrase at the end of the sequence
        seqs.append((PHRASES[txt[i]], txt[-1]))
    rigs = []
    for seq in seqs:
        rig_seq = [get_control_rig(seq[0][:, :, i])
                   for i in range(0, seq[0].shape[-1], SKIP_FRAMES_RATE)]
        rig_seq = filters(rig_seq)
        rigs.append(rig_seq)
    filters.reset()  # Reset filters after processing the phrase
    rigs = [pose for rig in rigs for pose in rig]
    for rig in rigs:
        data_queue.put(rig)
