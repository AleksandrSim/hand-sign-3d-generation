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


DATA_PATH = 'data/sequence/unified_data_master.npz'
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
            "Pinky Metacarpal R Ctrl", "Pinky 01 R Ctrl", "Pinky 02 R Ctrl", "Pinky 03 R Ctrl",
            "Wrist R Ctrl", "Lowerarm R Ctrl", "Upperarm R Ctrl"
            ]


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
                        payload[ctrl] = {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0}

                # print(payload)
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
    'Thumb 01 R Ctrl': ('RightFinger1Proximal', 'RightFinger1Metacarpal', 'RightFinger2Proximal'),
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
    'Thumb 01 R Ctrl': (0.0, -25.0, 10.0),
    'Index 01 R Ctrl': (0.0, 0.0, -14.0),
    'Index 02 R Ctrl': (0.0, 0.0, -14.0),
    'Index 03 R Ctrl': (0.0, 0.0, -3.5),
    'Middle 01 R Ctrl': (0.0, 0.0, -12.0),
    'Middle 02 R Ctrl': (0.0, 0.0, -16.0),
    'Middle 03 R Ctrl': (0.0, 0.0, -5.0),
    'Ring 01 R Ctrl': (0.0, 0.0, -14.0),
    'Ring 02 R Ctrl': (0.0, 0.0, -24.0),
    'Ring 03 R Ctrl': (0.0, 0.0, -14.0),
    'Pinky 01 R Ctrl': (0.0, -8.0, -12.0),
    'Pinky 02 R Ctrl': (0.0, 0.0, -20.0),
    'Pinky 03 R Ctrl': (0.0, 0.0, -8.0),
}


# Neutal pose for the arm itself (wrist, elbow, shoulder)
NEUTRAL_ARM: dict[str, tuple[float, float, float]] = {
    'Wrist R Ctrl': (0.0, 0.0, 0.0),
    'Lowerarm R Ctrl': (0.0, 0.0, 65.0),
    'Upperarm R Ctrl': (30.0, 0.0, 0.0)
}


def get_angle_by_3_points(p0: np.ndarray, p1: np.ndarray,
                          p2: np.ndarray) -> float:

    p0p1 = p0-p1
    p2p1 = p2-p1
    cosine_angle = np.dot(p0p1, p2p1) / \
        (np.linalg.norm(p0p1) * np.linalg.norm(p2p1))
    ang = np.degrees(np.arccos(cosine_angle)) - 180.0
    return float(ang)


def get_3d_angle(xyz: np.ndarray, keypoints: tuple[str, str, str])\
        -> tuple[float, float, float]:
    p0 = xyz[:, HAND_BONES.index(keypoints[0])]
    p1 = xyz[:, HAND_BONES.index(keypoints[1])]
    p2 = xyz[:, HAND_BONES.index(keypoints[2])]
    z = get_angle_by_3_points(p0, p1, p2)
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
        xyz = seq[:, :, i]
        angle, plane = get_hand_orientation(xyz)
        for ctrl, joints in CONTROLS.items():
            if len(joints) == 3:
                contol_rig[ctrl] = get_3d_angle(xyz, joints)
            # The last joint approximation
            elif len(joints) == 1:
                if joints[0] in contol_rig:
                    ref = contol_rig[joints[0]]
                    contol_rig[ctrl] = (ref[0]/2.0, ref[1]/2.0, ref[2]/2.0)
            if ctrl == 'Thumb 01 R Ctrl':
                rig = contol_rig[ctrl]
                rot = get_thumb_rotation(xyz, plane)
                # print(f'{ctrl} {rig[2]+180}, {rot}')
                contol_rig[ctrl] = (0.0, -rot/2,  0)
            # Take into account the neutral pose
            if ctrl in contol_rig and ctrl in NEUTRAL_POSE:
                neutral = NEUTRAL_POSE[ctrl]
                rig = contol_rig[ctrl]
                contol_rig[ctrl] = tuple(rig_i - neutral_i
                                         for rig_i, neutral_i
                                         in zip(rig, neutral))

        # Add the neutral arm pose
        for ctrl, joints in NEUTRAL_ARM.items():
            contol_rig[ctrl] = joints
            try:
                if ctrl == 'Wrist R Ctrl':
                    contol_rig[ctrl] = (
                        joints[0], joints[1]+angle[2],
                        joints[2])
            except:
                pass
        # print('Thumb 01 R Ctrl', contol_rig['Thumb 01 R Ctrl'])
        data_queue.put(contol_rig)
        get_hand_orientation(xyz)
        # break
    return data_queue


def get_thumb_rotation(xyz: np.ndarray, plane: np.ndarray) -> float:
    joints_idx = [HAND_BONES.index(p) for p in
                  ['RightFinger1Proximal', 'RightFinger1Metacarpal',
                   'RightFinger2Proximal', 'RightFinger2Metacarpal']]
    points = np.take(xyz, joints_idx, axis=1)  # (3, 4)
    print(points.shape)
    print(points)
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    normal = svd[0][:, -1]
    angle = np.degrees(np.arccos(
        np.dot(plane, normal)/(np.linalg.norm(plane)*np.linalg.norm(normal))))
    return float(angle)


def get_hand_orientation(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joints_idx = [HAND_BONES.index(p) for p in
                  ['RightFinger2Proximal', 'RightFinger3Metacarpal',
                   'RightFinger4Proximal']]
    xyz_hand = np.take(xyz, joints_idx, axis=1).T
    xyz_hand -= xyz_hand[1, :]
    plane = np.cross(xyz_hand[0], xyz_hand[2])
    normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    angle = get_plane_angle(plane, normals)
    # print(f'Angle: {angle}')
    return angle, plane


def get_plane_angle(plane: np.ndarray, normal: np.ndarray) -> np.ndarray:
    # Normals have unit length
    angle = np.degrees(np.arccos(np.dot(plane, normal)/np.linalg.norm(plane)))
    return angle


if __name__ == "__main__":
    # Define the URL where characters will be sent
    # endpoint_url = "http://127.0.0.1:8000"
    endpoint_url = "http://localhost:30010/remote/preset/RCP_Hand/function/Set%20Hand%20Pose"

    # Create a multiprocessing queue for communication between processes
    data_m_queue = multiprocessing.Queue()

    txt = ['A', 'B', 'A', 'B']
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
