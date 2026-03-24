import os
import pickle


def save_stub(stub_path, object):
    if stub_path is None:
        return

    parent_dir = os.path.dirname(stub_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(stub_path, 'wb') as f:
        pickle.dump(object, f)


def read_stub(read_from_stub, stub_path):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            object = pickle.load(f)
            return object 
    return None
