import pickle


def save_model(model):
    with open(str(model)+".pkl", 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(model):
    with open(str(model)+".pkl", 'rb') as f:
        model = pickle.load(f)
    return model