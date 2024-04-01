
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
        start_pose = np.array(rig_seq[0]['Wrist R Ctrl'])
        if self.prev_pos is not None:
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
