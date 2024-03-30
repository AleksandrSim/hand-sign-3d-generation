import time
import queue
import multiprocessing

import numpy as np
import requests

from src.control.config import ALL_CTRL
from scripts.retrive_data import filter_non_zero
from src.process_data.utils import char_index_map
from src.control.control_rig import get_control_rig

FPS = 30
FRAME_TIME = 1.0 / FPS
TIMEOUT = 0.1


DATA_PATH = 'data/sequence/unified_data_master.npz'
data = np.load(DATA_PATH)['data']


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
                break
                # values = generate_linear_interpolation((0., 0., 0.), 100)

                # for val in values:
                #     m_queue.put(val)
        except queue.Empty:  # Here queue is a mudule, not a class entity
            pass
        except Exception as e:
            print(f"Error occurred while sending data: {e}")


def listen_input(m_queue: multiprocessing.Queue):
    while True:
        try:
            user_input = input("Enter a string: ")
            for char in user_input:
                m_queue.put(char)
        except KeyboardInterrupt:
            break


def get_queued_data(txt: list[str]) -> multiprocessing.Queue:
    data_queue: multiprocessing.Queue = multiprocessing.Queue()
    assert len(txt) > 1, 'We can show >1 letter only'
    seqs = []
    for i in range(len(txt)-1):
        seqs.append(filter_non_zero(
            data[char_index_map[txt[i]], char_index_map[txt[i+1]], :, :, :]))
    seq = np.concatenate(seqs, axis=-1)
    for i in range(0, seq.shape[-1], 2):
        control_rig = get_control_rig(seq[:, :, i])
        data_queue.put(control_rig)
    return data_queue


if __name__ == "__main__":
    # Define the URL where characters will be sent
    # endpoint_url = "http://127.0.0.1:8000"
    endpoint_url = "http://localhost:30010/remote/preset/RCP_Hand/function/Set%20Hand%20Pose"

    # Create a multiprocessing queue for communication between processes
    data_m_queue = multiprocessing.Queue()

    txt = ['prob', 'A', 'L', 'E', 'K', 'S']
    # txt = ['prob', 'N', 'I', 'K', 'O', 'L', 'A', 'Y']
    # txt = ['prob', 'A', 'B', 'V', 'A']
    data_queue = get_queued_data(txt)

    # Create and start the processes

    sender_process = multiprocessing.Process(
        target=send_data, args=(data_queue, endpoint_url))
    sender_process.start()

    try:
        # Join the processes to wait for their completion
        # input_process.join()
        sender_process.join()
    except KeyboardInterrupt:
        # If the user interrupts the program, terminate the processes
        # input_process.terminate()
        sender_process.terminate()
        # If the user interrupts the program, terminate the processes
        # input_process.terminate()
        sender_process.terminate()
