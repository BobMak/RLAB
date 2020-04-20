import os
import pickle


def cache(filename, callback, cb_args=(), reset=False):
    if os.path.exists(filename) and not reset:
        with open(filename, 'rb') as f:

            return pickle.load(f)
    else:
        with open(filename, 'wb') as f:
            val = callback(*cb_args)
            pickle.dump(val, f, pickle.HIGHEST_PROTOCOL)
            return val