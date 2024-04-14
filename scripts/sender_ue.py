import time
import queue
import warnings
import threading
import multiprocessing

import numpy as np
import requests

from src.control.config import ALL_CTRL
from src.control.filters import Filters, interpolate_sequences
from src.control.sequences import PHRASES
from src.process_data.utils import char_index_map
from src.control.control_rig import get_control_rig
from src.process_data.sequence import filter_non_zero
from src.control.language_engine import transform_to_list

# Address of the Unreal Engine server
ADDRESS = "http://localhost:30010"
# Path to the endpoint in the Unreal Engine server
ENDPOINT = "/remote/preset/RCP_Hand/function/Set%20Hand%20Pose"
# Path to the file with the master sequence
DATA_PATH = "data/sequence/master_eng.npz"
FPS = 30
FRAME_TIME = 1.0 / FPS
# Rate of frames to skip (useful for speeding up sequences when FPS control
# is insufficient due to, e.g., limited compute power)
SKIP_FRAMES_RATE = 1
TIMEOUT = 0.1
SLEEP_TIME = 0.1  # Sleep time to check if we have a new input

filters = Filters()


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
            warnings.warn(f"Error occurred while sending data: {e}",
                          stacklevel=2)


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
        warnings.warn(f"{txt} is not a valid sequence. We support > 1 letters",
                      stacklevel=2)
        return
    seqs: list[tuple[np.ndarray, str]] = []
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


if __name__ == "__main__":
    data = np.load(DATA_PATH)['data']

    # Define the URL where characters will be sent
    endpoint_url = ADDRESS + ENDPOINT

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
        # Join the processes to wait for their compleDOtion
        input_thread.join()
        sender_process.join()

    except KeyboardInterrupt:
        # If the user interrupts the program, terminate the processes
        sender_process.terminate()
