# Collection of prerecorded sequences

import numpy as np

# TODO We need sequences of shape (3, 19, N), while the prerecorded data is
# in shape (19, 3, N). This should be updated with the data update.
PHRASES = {
    'name': np.load('data/sequence/Hello_my_name_is.npz'),
    'from': np.load('data/sequence/i_live_in.npz')
}

# TODO We need sequences of shape (3, 19, N), while the prerecorded data is
# in shape (19, 3, N). This should be updated with the data update.
PHRASES = {k: np.moveaxis(v['data'], 0, 1) for k, v in PHRASES.items()}
