import dynet_config as dy_conf
dy_conf.set(random_seed=2522620396)

import rnn
import utils
import csv
import dynet as dy
from time import time


def benchmark(func, repetitions  = 50):
    iterations = range(repetitions)
    ret = list()
    for i in iterations:
        print("Iteration %i" % (i+1))
        start = time()

        try:
            func()
            end = time()
            t = end - start
            ret.append(t)
        except:
            pass
    return ret



if __name__ == "__main__":

    p = "model.dy"
    dict_path = "vocab.txt"
    w2v_dict_path = "w2vvoc.txt"
    with open(dict_path) as f:
        missing_voc = {l[:-1]:i for i, l in enumerate(f)}

    with open(w2v_dict_path) as f:
        w2v_voc = {l[:-1]:i for i, l in enumerate(f)}

    pc = dy.ParameterCollection()

    W = pc.load_param(p, '/W')
    V = pc.load_param(p, '/V')
    b = pc.load_param(p, '/b')

    w2v_emb = pc.load_lookup_param(p, '/w2v-wemb')
    missing_emb = pc.load_lookup_param(p, '/missing-wemb')

    VOC_SIZE = len(missing_voc)
    WEM_DIMENSIONS = 100

    NUM_LAYERS = 1
    HIDDEN_DIM = 20

    FF_HIDDEN_DIM = 10

    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, pc)
    builder.param_collection().populate(p, "/vanilla-lstm-builder/")

    elements = rnn.ModelElements(W, V, b, w2v_emb, missing_emb, pc, builder)

    embeddings_index = utils.WordEmbeddingIndex(w2v_emb, w2v_voc, missing_emb, missing_voc)

    with open("resources/concurring_reach_stmts.tsv") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        rows = [utils.Instance.normalize(s[5].strip('"')) for s in reader]


    def worker():
        for row in rows:
            try:
                pred = rnn.run_instance(row, elements, embeddings_index)
                pred.value()
            except:
                pass

    times = benchmark(worker, 50)

    print("%s\n" % ','.join(str(t) for t in times))



