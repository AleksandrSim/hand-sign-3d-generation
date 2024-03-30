import numpy as np

from src.control.config import CONTROLS, NEUTRAL_POSE
from scripts.retrive_data import HAND_BONES
from src.control.geometry import get_ort_plane, angle_between_vectors, \
    get_angle_by_3_points, project_vector_onto_plane


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


def get_thumb_rotation(xyz: np.ndarray, plane: np.ndarray)\
        -> tuple[float, float]:
    p0 = xyz[:, HAND_BONES.index('RightFinger1Proximal')]
    p1 = xyz[:, HAND_BONES.index('RightFinger1Metacarpal')]
    p2 = xyz[:, HAND_BONES.index('RightFinger2Proximal')]
    p3 = xyz[:, HAND_BONES.index('RightFinger2Metacarpal')]

    index_bone = p2 - p3
    thumb_bone = p0 - p1
    v_plane = get_ort_plane(plane, index_bone)
    thumb_bone_proj = project_vector_onto_plane(thumb_bone, plane)
    thumb_bone_proj_v = project_vector_onto_plane(thumb_bone, v_plane)
    r2 = angle_between_vectors(index_bone, thumb_bone_proj_v)
    r3 = angle_between_vectors(index_bone, thumb_bone_proj)
    return r2, r3


def get_finger_angle(xyz: np.ndarray, keypoints: tuple[str, str, str],
                     plane: np.ndarray)\
        -> tuple[float, float, float]:
    r2 = 0.0
    p0 = xyz[:, HAND_BONES.index(keypoints[0])]
    p1 = xyz[:, HAND_BONES.index(keypoints[1])]
    p2 = xyz[:, HAND_BONES.index(keypoints[2])]
    # Apply to proximal joint of the selected fingers, which can hav lateral
    # movement
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


def get_control_rig(xyz: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Get the control rig angles from a given set of keypoints.

    Args:
        xyz (np.ndarray): The keypoints of the hand. Shape is (3, 19).

    Returns:
        dict[str, tuple[float, float, float]]: The resulted control rig angles.
    """
    control_rig: dict[str, tuple[float, float, float]] = {}
    angle, plane = get_hand_orientation(xyz)
    for ctrl, joints in CONTROLS.items():
        if len(joints) == 3:
            control_rig[ctrl] = get_finger_angle(xyz, joints, plane)
        # The last joint approximation
        elif len(joints) == 1:
            if joints[0] in control_rig:
                ref = control_rig[joints[0]]
                control_rig[ctrl] = (ref[0]/2.0, ref[1]/2.0, ref[2]/2.0)

    r2, r3 = get_thumb_rotation(xyz, plane)
    control_rig['Thumb 01 R Ctrl'] = (0.0, -r2, -r3)

    # Add the wrist pose, it defines the whole hand orientation
    control_rig['Wrist R Ctrl'] = (angle[0], angle[1], angle[2])

    # Take into account the neutral pose
    for ctrl, neutral in NEUTRAL_POSE.items():
        if ctrl in control_rig:
            pose = control_rig[ctrl]
            control_rig[ctrl] = tuple(pose_i - neutral_i
                                      for pose_i, neutral_i
                                      in zip(pose, neutral))
        else:
            control_rig[ctrl] = neutral

    return control_rig
