import time
import queue
import multiprocessing

import numpy as np
import requests

from scripts.retrive_data import HAND_BONES, filter_non_zero
from src.process_data.utils import char_index_map, letter_to_index

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


# Neutral pose for the hand and the arm
NEUTRAL_POSE: dict[str, tuple[float, float, float]] = {
    'Thumb 01 R Ctrl': (0.0, -20.0, 10.0),
    'Index 01 R Ctrl': (0.0, 0.0, -14.0),
    'Index 02 R Ctrl': (0.0, 7.0, -14.0),
    'Index 03 R Ctrl': (0.0, 0.0, -3.5),
    'Middle 01 R Ctrl': (0.0, 0.0, -12.0),
    'Middle 02 R Ctrl': (0.0, 7.0, -16.0),
    'Middle 03 R Ctrl': (0.0, 0.0, -5.0),
    'Ring 01 R Ctrl': (0.0, 0.0, -14.0),
    'Ring 02 R Ctrl': (0.0, 0.0, -24.0),
    'Ring 03 R Ctrl': (0.0, 0.0, -14.0),
    'Pinky 01 R Ctrl': (0.0, -8.0, -12.0),
    'Pinky 02 R Ctrl': (0.0, 0.0, -20.0),
    'Pinky 03 R Ctrl': (0.0, 0.0, -8.0),
    'Wrist R Ctrl': (0.0, 0.0, 0.0),
    'Lowerarm R Ctrl': (0.0, 0.0, 65.0),
    'Upperarm R Ctrl': (30.0, 0.0, 0.0)
}


def get_angle_by_3_points(p0: np.ndarray, p1: np.ndarray,
                          p2: np.ndarray) -> float:

    p0p1 = p0-p1
    p2p1 = p2-p1
    ang = angle_between_vectors(p0p1, p2p1) - 180.0
    return float(ang)


def get_ort_plane(plane: np.ndarray, vect: np.ndarray) -> np.ndarray:
    """Get the ortogonal plane to a given plane, passing through a vector

    Args:
        plane (np.ndarray): Plane, defined by its normal.
        vect (np.ndarray): Vector (3 elements).

    Returns:
        np.ndarray: The resulted plane normal.
    """
    norm = np.cross(vect, plane)
    return norm/np.linalg.norm(norm)


def project_vector_onto_plane(vect: np.ndarray, plane: np.ndarray)\
        -> np.ndarray:
    """Project a vector onto a plane.

    https://math.stackexchange.com/questions/633181/formula-to-project-a-vector-onto-a-plane#

    Args:
        vect (np.ndarray): Vector to be projected (as 3 values).
        plane (np.ndarray): Plane, defined by its norm.

    Returns:
        np.ndarray: The projection of the vector onto the plane.
    """
    proj = vect - np.dot(vect, plane)*plane
    return proj


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Get the angle between two vectors, assuming the vectors are not
    collinear and have non-zero length.

    Args:
        v1 (np.ndarray): First vector (as 3 values).
        v2 (np.ndarray): Second vector (as 3 values).

    Returns:
        float: The angle in degrees.
    """
    return float(np.degrees(np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))))


def get_finger_angle(xyz: np.ndarray, keypoints: tuple[str, str, str],
                     plane: np.ndarray)\
        -> tuple[float, float, float]:
    r2 = 0.0
    p0 = xyz[:, HAND_BONES.index(keypoints[0])]
    p1 = xyz[:, HAND_BONES.index(keypoints[1])]
    p2 = xyz[:, HAND_BONES.index(keypoints[2])]
    # Apply to proxima joint of the selected fingers
    if 'Proximal' in keypoints[1] and any(n in keypoints[1]
                                          for n in ['2', '3', '4', '5']):
        base_bone = p1 - p0
        finger_bone = p2 - p1
        v_plane = get_ort_plane(plane, base_bone)
        finger_bone_proj = project_vector_onto_plane(finger_bone, plane)
        finger_bone_proj_v = project_vector_onto_plane(finger_bone, v_plane)
        r3 = -angle_between_vectors(base_bone, finger_bone_proj_v)
        # The last term is to deal with edge cases due to unmerical instability
        r2 = -angle_between_vectors(base_bone, finger_bone_proj)\
            * (90.0 - abs(r3)) / 90.0
        print(keypoints, r2, r3)
    else:
        r3 = get_angle_by_3_points(p0, p1, p2)
    return 0.0, r2, r3


def get_queued_data(txt: list[str]) -> multiprocessing.Queue:
    data_queue: multiprocessing.Queue = multiprocessing.Queue()
    assert len(txt) > 1, 'We can show >1 letter only'
    seqs = []
    for i in range(len(txt)-1):
        seqs.append(filter_non_zero(
            data[char_index_map[txt[i]], char_index_map[txt[i+1]], :, :, :]))
    seq = np.concatenate(seqs, axis=-1)
    for i in range(0, seq.shape[-1], 2):
        contol_rig = {}
        xyz = seq[:, :, i]
        angle, plane = get_hand_orientation(xyz)
        for ctrl, joints in CONTROLS.items():
            if len(joints) == 3:
                contol_rig[ctrl] = get_finger_angle(xyz, joints, plane)
            # The last joint approximation
            elif len(joints) == 1:
                if joints[0] in contol_rig:
                    ref = contol_rig[joints[0]]
                    contol_rig[ctrl] = (ref[0]/2.0, ref[1]/2.0, ref[2]/2.0)
            if ctrl == 'Thumb 01 R Ctrl':
                r2, r3 = get_thumb_rotation(xyz, plane)
                contol_rig[ctrl] = (0.0, -r2, -r3)

        # Add the wrist pose
        contol_rig['Wrist R Ctrl'] = (angle[0], angle[1], angle[2])

        # Take into account the neutral pose
        for ctrl, neutral in NEUTRAL_POSE.items():
            if ctrl in contol_rig:
                pose = contol_rig[ctrl]
                contol_rig[ctrl] = tuple(pose_i - neutral_i
                                         for pose_i, neutral_i
                                         in zip(pose, neutral))
            else:
                contol_rig[ctrl] = neutral
        data_queue.put(contol_rig)
        get_hand_orientation(xyz)
        # break
    return data_queue


def get_thumb_rotation(xyz: np.ndarray, plane: np.ndarray)\
        -> tuple[float, float]:
    # joints_idx = [HAND_BONES.index(p) for p in
    #               ['RightFinger1Proximal', 'RightFinger1Metacarpal',
    #                'RightFinger2Proximal', 'RightFinger2Metacarpal']]
    # points = np.take(xyz, joints_idx, axis=1)  # (3, 4)
    # svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    # normal = svd[0][:, -1]
    # plane_xz = plane.copy()
    # plane_xz = np.array([plane_xz[0], 0.0, plane_xz[2]])  # XZ projection
    # normal_xz = np.array([normal[0], 0.0, normal[2]])  # XZ projection
    # angle = np.degrees(np.arccos(
    #     np.dot(plane_xz, normal_xz)/(np.linalg.norm(plane_xz)*np.linalg.norm(normal_xz))))
    # if angle > 90.0:
    #     angle = 180.0 - angle

    p0 = xyz[:, HAND_BONES.index('RightFinger1Proximal')]
    p1 = xyz[:, HAND_BONES.index('RightFinger1Metacarpal')]
    p2 = xyz[:, HAND_BONES.index('RightFinger2Proximal')]
    p3 = xyz[:, HAND_BONES.index('RightFinger2Metacarpal')]

    index_bone = p2-p3
    thumb_bone = p0 - p1
    v_plane = get_ort_plane(plane, index_bone)
    thumb_bone_proj = project_vector_onto_plane(thumb_bone, plane)
    thumb_bone_proj_v = project_vector_onto_plane(thumb_bone, v_plane)
    r2 = angle_between_vectors(index_bone, thumb_bone_proj_v)
    r3 = angle_between_vectors(index_bone, thumb_bone_proj)
    return r2, r3


def get_hand_orientation(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joints_idx = [HAND_BONES.index(p) for p in
                  ['RightFinger2Proximal', 'RightFinger3Metacarpal',
                   'RightFinger5Proximal']]
    xyz_hand = np.take(xyz, joints_idx, axis=1).T
    xyz_hand -= xyz_hand[1, :]  # Define the origin of the hand
    plane = np.cross(xyz_hand[0], xyz_hand[2])
    plane = plane / np.linalg.norm(plane)
    plane_yz = np.array([0.0, plane[1], plane[2]])
    plane_xz = np.array([plane[0], 0.0, plane[2]])
    plane_xy = np.array([plane[0], plane[1], 0.0])
    angles = np.array((
        angle_between_vectors(plane_xy, np.array([0, 1, 0])),  # yaw
        angle_between_vectors(plane_yz, np.array([0, 0, 1])),  # pitch
        angle_between_vectors(plane_xz, np.array([0, 0, 1]))))  # roll
    angles[angles > 90.0] = (180.0 - angles)[angles > 90.0]
    return angles, plane


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
