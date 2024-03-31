# Collection of prerecorded sequences

import numpy as np

# TODO We need sequences of shape (3, 19, N), while the prerecorded data is
# in shape (19, 3, N). This should be updated with the data update.
PHRASES = {
    'name': np.moveaxis(
        np.load('data/sequence/Hello_my_name_is.npz')['data'], 0, 1),
    'from': np.moveaxis(np.load('data/sequence/i_live_in.npz')['data'], 0, 1)
}
