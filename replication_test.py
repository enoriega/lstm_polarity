import dynet_config as dy_conf
dy_conf.set(random_seed=2522620396)

from rnn import *
from utils import *





import dynet as dy

if __name__ == "__main__":

    sentence = "To formally prove that increased ROS levels enhance anti-tumour effects of the SG-free diet , the authors crossed Emu-Myc mice with mice deficient for Tigar , a fructose-2 ,6-bisphosphatase , which limits glycolysis and favours pentose phosphate pathways , thus limiting ROS levels XREF_BIBR , XREF_BIBR ( XREF_FIG ) ."

    p = "/Users/enrique/scratch/lstm_polarity/model.dy"
    dictPath = "/Users/enrique/scratch/lstm_polarity/vocab.txt"
    w2vDictPath = "/Users/enrique/scratch/lstm_polarity/w2vvoc.txt"

    with open(dictPath) as f:
        missing_voc = {v[:-1]:i for i, v in enumerate(f)}

    with open(w2vDictPath) as f:
        w2v_voc = {v[:-1]: i for i, v in enumerate(f)}

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

    elements = ModelElements(W, V, b, w2v_emb, missing_emb, pc, builder)

    embeddings_index = WordEmbeddingIndex(w2v_emb, w2v_voc, missing_emb, missing_voc)

    tokens = Instance.normalize(sentence)

    pred = run_instance(tokens, elements, embeddings_index)

    print(pred.value())
