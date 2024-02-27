


import time
import queue
import multiprocessing

import requests
import numpy as np

from scripts.retrive_data import filter_non_zero

FPS = 30
FRAME_TIME = 1.0 / FPS
TIMEOUT = 0.1


def send_data(url: str, data_path: str, transition: list[int] ):          
    data = np.load(data_path)['data']
    seq = data[transition[0], transition[1], :,:,:]
    seq = filter_non_zero(seq)
    
    idx = 0
    frames = seq.shape[-1]
    print(frames)

    while idx < frames:
        start_t = time.perf_counter()
        requests.put(url, json={
            "Parameters": {
                "Thumb 01 R Ctrl": {
                    "Pitch": seq[0, 0, idx],
                    "Yaw": seq[1, 0, idx],
                    "Roll": seq[2, 0, idx]
                },
                "Thumb 02 R Ctrl": {
                    "Pitch": seq[0, 1, idx],
                    "Yaw": seq[1, 1, idx],
                    "Roll": seq[2, 1, idx]
                },
                "Thumb 03 R Ctrl": {
                    "Pitch": seq[0, 2, idx],
                    "Yaw": seq[1, 2, idx],
                    "Roll": seq[2, 2, idx]
                },
                "Index Metacarpal R Ctrl": {
                    "Pitch": seq[0, 3, idx],
                    "Yaw": seq[1, 3, idx],
                    "Roll": seq[2, 3, idx]
                },
                "Index 01 R Ctrl": {
                    "Pitch": seq[0, 4, idx],
                    "Yaw": seq[1, 4, idx],
                    "Roll": seq[2, 4, idx]
                },
                "Index 02 R Ctrl": {
                    "Pitch": seq[0, 5, idx],
                    "Yaw": seq[1, 5, idx],
                    "Roll": seq[2, 5, idx]
                },
                "Index 03 R Ctrl": {
                    "Pitch": seq[0, 6, idx],
                    "Yaw": seq[1, 6, idx],
                    "Roll": seq[2, 6, idx]
                },
                "Middle Metacarpal R Ctrl": {
                    "Pitch": seq[0, 7, idx],
                    "Yaw": seq[1, 7, idx],
                    "Roll": seq[2, 7, idx]
                },
                "Middle 01 R Ctrl": {
                    "Pitch": seq[0, 8, idx],
                    "Yaw": seq[1, 8, idx],
                    "Roll": seq[2, 8, idx]
                },
                "Middle 02 R Ctrl": {
                    "Pitch": seq[0, 9, idx],
                    "Yaw": seq[1, 9, idx],
                    "Roll": seq[2, 9, idx]
                },
                "Middle 03 R Ctrl": {
                    "Pitch": seq[0, 10, idx],
                    "Yaw": seq[1, 10, idx],
                    "Roll": seq[2, 10, idx]
                },
                "Ring Metacarpal R Ctrl": {
                    "Pitch": seq[0, 11, idx],
                    "Yaw": seq[1, 11, idx],
                    "Roll": seq[2, 11, idx]
                },
                "Ring 01 R Ctrl": {
                    "Pitch": seq[0, 12, idx],
                    "Yaw": seq[1, 12, idx],
                    "Roll": seq[2, 12, idx]
                },
                "Ring 02 R Ctrl": {
                    "Pitch": seq[0, 13, idx],
                    "Yaw": seq[1, 13, idx],
                    "Roll": seq[2, 13, idx]
                },
                "Ring 03 R Ctrl": {
                    "Pitch": seq[0, 14, idx],
                    "Yaw": seq[1, 14, idx],
                    "Roll": seq[2, 14, idx]
                },
                "Pinky Metacarpal R Ctrl": {
                    "Pitch": seq[0, 15, idx],
                    "Yaw": seq[1, 15, idx],
                    "Roll": seq[2, 15, idx]
                },
                "Pinky 01 R Ctrl": {
                    "Pitch": seq[0, 16, idx],
                    "Yaw": seq[1, 16, idx],
                    "Roll": seq[2, 16, idx]
                },
                "Pinky 02 R Ctrl": {
                    "Pitch": seq[0, 17, idx],
                    "Yaw": seq[1, 17, idx],
                    "Roll": seq[2, 17, idx]
                },
                "Pinky 03 R Ctrl": {
                    "Pitch": seq[0, 18, idx],
                    "Yaw": seq[1, 18, idx],
                    "Roll": seq[2, 18, idx]
                }
            },
            "generateTransaction": True
        },
        timeout=1.0)
        send_time = time.perf_counter() - start_t
        time.sleep(max(FRAME_TIME-send_time, 0.0))
        idx +=1
        print(idx)
            

            


def listen_input(m_queue: multiprocessing.Queue):
    while True:
        try:
            user_input = input("Enter a string: ")
            for char in user_input:
                m_queue.put(char)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    # Define the URL where characters will be sent
    # endpoint_url = "http://127.0.0.1:8000"
    endpoint_url = "http://localhost:30010/remote/preset/RCP_Hands/function/Set%20Hand%20Pose"
    data = np.load('/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_reverse_inc.npz')['data']
    send_data(endpoint_url, '/Users/aleksandrsimonyan/Desktop/complete_sequence/unified_data_reverse_inc.npz', [0,1])