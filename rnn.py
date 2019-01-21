import dynet as dy
from collections import namedtuple

ModelElements = namedtuple("ModelElements", "W V b w2v_emb param_collection builder")


def run_instance(tokens, polarity, model_elems, embeddings):

    # Renew the computational graph
    dy.renew_cg()

    builder = model_elems.builder
    V = model_elems.V
    W = model_elems.W
    b = model_elems.b

    # Fetch the embeddings for the current sentence
    words = tokens
    inputs = [embeddings[w] for w in words]

    # Run FF over the LSTM
    lstm = builder.initial_state()
    outputs = lstm.transduce(inputs)

    # Get the last embedding
    selected = outputs[-1]

    # Concatenate the polarity bit to the selected vector
    prediction_input = dy.concatenate([selected, dy.scalarInput(1 if polarity else 0)])
    #prediction_input = selected

    # Run the FF network for classification
    prediction = dy.logistic(V * (W * prediction_input + b))

    return prediction


def prediction_loss(instance, prediction):
    # Compute the loss
    y_true = dy.scalarInput(1 if instance.polarity else 0)
    loss = dy.binary_log_loss(prediction, y_true)

    return loss


def build_model(w2v_embeddings):
    WEM_DIMENSIONS = 100

    NUM_LAYERS = 1
    HIDDEN_DIM = 20

    FF_HIDDEN_DIM = 10

    params = dy.ParameterCollection()
    w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, init=w2v_embeddings.matrix, name="w2v-wemb")

    # Feed-Forward parameters
    W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
    #W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM), name="W")
    b = params.add_parameters((FF_HIDDEN_DIM), name="b")
    V = params.add_parameters((1, FF_HIDDEN_DIM), name="V")

    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, params)

    ret = ModelElements(W, V, b, w2v_wemb, params, builder)

    return ret

