import numpy as np


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
