# Collection of prerecorded sequences

import numpy as np

# Fill the dictionary with prerecorded sequences, for example:
#      'name': np.load('data/sequence/Hello_my_name_is.npz'),
PHRASES: dict[str, np.ndarray] = {
}

# TODO We need sequences of shape (3, 20, N), while the prerecorded data is
# in shape (20, 3, N). This should be updated with the data update.
PHRASES = {k: np.moveaxis(v['data'], 0, 1) for k, v in PHRASES.items()}
