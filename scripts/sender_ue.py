import time
import queue
import multiprocessing

import numpy as np
import requests

from scripts.retrive_data import HAND_BONES, filter_non_zero
from src.process_data.utils import letter_to_index

FPS = 30
FRAME_TIME = 1.0 / FPS
TIMEOUT = 0.1


DATA_PATH = 'data/sequence/unified_data_reverse_inc.npz'
data = np.load(DATA_PATH)['data']


def generate_linear_interpolation(
        extr: tuple[float, float, float],
        N: int) -> list[list[float]]:
    # Calculate the step size for interpolation
    steps = [x / (N // 2) for x in extr]

    values = [[step * i for step in steps] for i in range(N // 2)]
    values += [[extr[n] - steps[n] * i for n in range(len(extr))]
               for i in range(N // 2 + 1)]

    return values


ALL_CTRL = ["Thumb 01 R Ctrl", "Thumb 02 R Ctrl", "Thumb 03 R Ctrl",
            "Index Metacarpal R Ctrl", "Index 01 R Ctrl", "Index 02 R Ctrl", "Index 03 R Ctrl",
            "Middle Metacarpal R Ctrl", "Middle 01 R Ctrl", "Middle 02 R Ctrl", "Middle 03 R Ctrl",
            "Ring Metacarpal R Ctrl", "Ring 01 R Ctrl", "Ring 02 R Ctrl", "Ring 03 R Ctrl",
            "Pinky Metacarpal R Ctrl", "Pinky 01 R Ctrl", "Pinky 02 R Ctrl", "Pinky 03 R Ctrl"
            ]


def send_data(m_queue: multiprocessing.Queue, url: str):
    while True:
        try:
            if not m_queue.empty():
                start_t = time.perf_counter()
                s = m_queue.get(timeout=TIMEOUT)
                payload = {}
                print(s)
                for ctrl in ALL_CTRL:
                    if ctrl in s:
                        angle = s[ctrl]
                        payload[ctrl] = {"Roll": angle[0],
                                         "Pitch": angle[1],
                                         "Yaw": angle[2]}
                    else:
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


# Define how to canculate angles for each control
CONTROLS = {
    'Thumb 01 R Ctrl': ('RightFinger2Proximal', 'RightFinger1Metacarpal', 'RightFinger1Proximal'),
    'Thumb 02 R Ctrl': ('RightFinger1Metacarpal', 'RightFinger1Proximal', 'RightFinger1Distal'),
    'Thumb 03 R Ctrl': ('Thumb 02 R Ctrl', ),
    'Index 01 R Ctrl': ('RightFinger2Metacarpal', 'RightFinger2Proximal', 'RightFinger2Medial'),
    'Index 02 R Ctrl': ('RightFinger2Proximal', 'RightFinger2Medial', 'RightFinger2Distal'),
    'Index 03 R Ctrl': ('Index 02 R Ctrl', ),
    'Middle 01 R Ctrl': ('RightFinger3Metacarpal', 'RightFinger3Proximal', 'RightFinger3Medial'),
    'Middle 02 R Ctrl': ('RightFinger3Proximal', 'RightFinger3Medial', 'RightFinger3Distal'),
    'Middle 03 R Ctrl': ('Middle 02 R Ctrl', ),
    'Ring 01 R Ctrl': ('RightFinger4Metacarpal', 'RightFinger4Proximal', 'RightFinger4Medial'),
    'Ring 02 R Ctrl': ('RightFinger4Proximal', 'RightFinger4Medial', 'RightFinger4Distal'),
    'Ring 03 R Ctrl': ('Ring 02 R Ctrl', ),
    'Pinky 01 R Ctrl': ('RightFinger5Metacarpal', 'RightFinger5Proximal', 'RightFinger5Medial'),
    'Pinky 02 R Ctrl': ('RightFinger5Proximal', 'RightFinger5Medial', 'RightFinger5Distal'),
    'Pinky 03 R Ctrl': ('Pinky 02 R Ctrl', ),
}

NEUTRAL_POSE: dict[str, tuple[float, float, float]] = {
    'Index 01 R Ctrl': (0.0, 0.0, -14.0),
    'Index 02 R Ctrl': (0.0, 0.0, -7.0),
    'Index 03 R Ctrl': (0.0, 0.0, -3.5),
    'Middle 01 R Ctrl': (0.0, 0.0, -12.0),
    'Middle 02 R Ctrl': (0.0, 0.0, -16.0),
    'Middle 03 R Ctrl': (0.0, 0.0, -5.0)
}


def get_angles_by3points(xyz: np.ndarray, keypoints: tuple[str, str, str]):
    p0 = xyz[:, HAND_BONES.index(keypoints[0])]
    p1 = xyz[:, HAND_BONES.index(keypoints[1])]
    p2 = xyz[:, HAND_BONES.index(keypoints[2])]
    p0p1 = p0-p1
    p2p1 = p2-p1
    cosine_angle = np.dot(p0p1, p2p1) / \
        (np.linalg.norm(p0p1) * np.linalg.norm(p2p1))
    z = np.degrees(np.arccos(cosine_angle)) - 180.0
    y = 0.0
    return 0.0, y, z


def get_queued_data(txt: list[str]) -> multiprocessing.Queue:
    data_queue: multiprocessing.Queue = multiprocessing.Queue()
    assert len(txt) > 1, 'We can show >1 letter only'
    seqs = []
    for i in range(len(txt)-1):
        seqs.append(filter_non_zero(
            data[letter_to_index[txt[i]], letter_to_index[txt[i+1]], :, :, :]))
    seq = np.concatenate(seqs, axis=-1)
    for i in range(seq.shape[-1]):
        contol_rig = {}
        for ctrl, joints in CONTROLS.items():
            print(len(joints))
            if len(joints) == 3:
                contol_rig[ctrl] = get_angles_by3points(seq[:, :, i], joints)
            # The last joint approximation
            elif len(joints) == 1:
                if joints[0] in contol_rig:
                    ref = contol_rig[joints[0]]
                    contol_rig[ctrl] = (ref[0]/2.0, ref[1]/2.0, ref[2]/2.0)
            # Take into account the neutral pose
            if ctrl in contol_rig and ctrl in NEUTRAL_POSE:
                neutral = NEUTRAL_POSE[ctrl]
                rig = contol_rig[ctrl]
                contol_rig[ctrl] = tuple(rig_i - neutral_i
                                         for rig_i, neutral_i
                                         in zip(rig, neutral))
            # TODO Add 2 lines angle with intersection
            if ctrl == 'Thumb 01 R Ctrl':
                rig = contol_rig[ctrl]
                contol_rig[ctrl] = (0.0, 0.0, (rig[2]+180)/2.0 - 27.0)
        data_queue.put(contol_rig)

    return data_queue


if __name__ == "__main__":
    # Define the URL where characters will be sent
    # endpoint_url = "http://127.0.0.1:8000"
    endpoint_url = "http://localhost:30010/remote/preset/RCP_Hands/function/Set%20Hand%20Pose"

    # Create a multiprocessing queue for communication between processes
    data_m_queue = multiprocessing.Queue()

    txt = ['A', 'B', 'V', 'A']
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
