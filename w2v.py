import gzip, pickle
import numpy as np


class W2VEmbeddings:

    def __init__(self, raw):
        self.keys = set(raw.keys())
        self.keys.remove("1579375")  # Had to remove it manually
        self.voc = {w:ix for ix, w in enumerate(sorted(list(self.keys)))}
        arrays = [raw[k] for k in sorted(self.keys)]

        self.matrix = np.vstack(arrays)

    def __contains__(self, item):
        return item in self.keys

    def __getitem__(self, item):
        return self.voc[item]

    def shape(self):
        return self.matrix.shape


def load_embeddings(path):

    with gzip.open(path, "r") as f:
        raw = pickle.load(f)

    return W2VEmbeddings(raw)

