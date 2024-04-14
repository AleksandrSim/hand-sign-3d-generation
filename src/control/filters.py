from abc import ABC, abstractmethod

import numpy as np


class Filter(ABC):
    @abstractmethod
    def reset(self) -> None:
        # Reset the filter internal state
        pass

    @abstractmethod
    def __call__(self, rig_seq: list[dict[str, tuple[float, float, float]]])\
            -> list[dict[str, tuple[float, float, float]]]:
        return rig_seq


class WristSmoother(Filter):
    def __init__(self):
        self.prev_pos: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_pos = None

    def __call__(self, rig_seq: list[dict[str, tuple[float, float, float]]])\
            -> list[dict[str, tuple[float, float, float]]]:
        if len(rig_seq) > 0:
            if self.prev_pos is not None:
                start_pose = np.array(rig_seq[0]['Wrist R Ctrl'])
                d_pose = self.prev_pos - start_pose
                n_steps = len(rig_seq)
                for i, pose in enumerate(rig_seq):
                    if 'Wrist R Ctrl' in pose:
                        # Apply linear offset interpolation
                        rig_seq[i]['Wrist R Ctrl']\
                            = tuple(d_pose * (n_steps - i) / n_steps
                                    + rig_seq[i]['Wrist R Ctrl'])
            self.prev_pos = np.array(rig_seq[-1]['Wrist R Ctrl'])
        return rig_seq


class Filters(Filter):
    def __init__(self) -> None:
        self.filters = [
            WristSmoother(),
        ]

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

    def __call__(self, rig_seq: list[dict[str, tuple[float, float, float]]])\
            -> list[dict[str, tuple[float, float, float]]]:
        for f in self.filters:
            rig_seq = f(rig_seq)
        return rig_seq


def linear_interpolate(start: np.ndarray, end: np.ndarray, num_steps: int)\
        -> np.ndarray:
    # This function create interpolated frames between start and end frames
    return np.linspace(start, end, num_steps, axis=2)


def interpolate_sequences(data, start_idx: int, end_idx: int,
                          next_idx: int | None = None):
    transition_data = data[start_idx, end_idx, :, :, :]
    non_zero_frames_mask = np.any(transition_data != 0, axis=(0, 1))
    transition_data = transition_data[:, :, non_zero_frames_mask]

    if next_idx:
        next_transition_data = data[end_idx, next_idx, :, :, :]
        start_frame = transition_data[:, :, -1:]
        end_frame = next_transition_data[:, :, :1]
        interpolated_frames = linear_interpolate(
            start_frame, end_frame, num_steps=170)
        interpolated_frames_squeezed = np.squeeze(interpolated_frames, axis=-1)
        transition_data = np.concatenate(
            (transition_data[:, :, :], interpolated_frames_squeezed), axis=2)
    return transition_data
