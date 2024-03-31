import time
import queue
import warnings
import threading
import multiprocessing

import numpy as np
import requests

from src.control.config import ALL_CTRL
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
    seqs = []
    for i in range(len(txt)-1):
        if txt[i] in PHRASES:
            seqs.append(PHRASES[txt[i]])
        elif txt[i] in char_index_map and txt[i+1] in char_index_map:
            seqs.append(filter_non_zero(
                data[char_index_map[txt[i]],
                     char_index_map[txt[i+1]], :, :, :]))
    if txt[-1] in PHRASES:
        # Deal with the case when a keyphrase at the end of the sequence
        seqs.append(PHRASES[txt[i]])
    seq = np.concatenate(seqs, axis=-1)
    for i in range(0, seq.shape[-1], SKIP_FRAMES_RATE):
        control_rig = get_control_rig(seq[:, :, i])
        data_queue.put(control_rig)


if __name__ == "__main__":
    # Define the URL where characters will be sent
    # endpoint_url = "http://127.0.0.1:8000"
    endpoint_url = "http://localhost:30010/remote/preset/RCP_Hand/function/Set%20Hand%20Pose"

    # Create a multiprocessing queue for communication between processes
    data_queue = multiprocessing.Queue()

    # Create and start the processes
    input_thread = threading.Thread(
        target=listen_input, args=(data_queue,))
    input_thread.daemon = True
    sender_process = multiprocessing.Process(
        target=send_data, args=(data_queue, endpoint_url))
    input_thread.start()
    sender_process.start()

    try:
        # Join the processes to wait for their completion
        input_thread.join()
        sender_process.join()

    except KeyboardInterrupt:
        # If the user interrupts the program, terminate the processes
        sender_process.terminate()
