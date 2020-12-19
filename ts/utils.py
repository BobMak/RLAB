import os
import pickle


a = [0.232]

with open('save.pkl', 'wb') as f:
    pickle.dump(a, f)

with open('save.pkl', 'rb') as f:
    model = pickle.load(f)


def cache(filename, callback, cb_args=(), reset=False):
    if os.path.exists(filename) and not reset:
        with open(filename, 'rb') as f:

            return pickle.load(f)
    else:
        with open(filename, 'wb') as f:
            val = callback(*cb_args)
            pickle.dump(val, f, pickle.HIGHEST_PROTOCOL)
            return val