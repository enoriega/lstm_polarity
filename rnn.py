import dynet as dy
from collections import namedtuple
import numpy as np
from utils import *

ModelElements = namedtuple("ModelElements", "W V b w2v_emb missing_emb param_collection builder")


def run_instance(instance, model_elems, embeddings):

    # Renew the computational graph
    dy.renew_cg()

    builder = model_elems.builder
    builder.set_dropouts(0.2, 0.2)
    
    W = model_elems.W
    V = model_elems.V
    b = model_elems.b
    
    collected_vectors = list()

    for segment in instance.get_segments():

        if len(segment) > 0:

            # Fetch the embeddings for the current sentence
            inputs = [embeddings[w] for w in segment]
            
            #print(inputs[-1].npvalue())
            #input('press enter to continue')

            # Run FF over the LSTM
            lstm = builder.initial_state()
            outputs = lstm.transduce(inputs)

            # Get the last embedding
            selected = outputs[-1]

            # Collect it
            collected_vectors.append(selected)
        else:
            zero_vector = dy.zeros(20)
            collected_vectors.append(zero_vector)

    # Concatenate the selected vectors and the polarity trigger feature
    lstm_result = dy.concatenate(collected_vectors)  # shape=4*20

    trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

    ff_input = dy.concatenate([trigger_expression, lstm_result])

    # Run the FF network for classification
    prediction = dy.logistic(V * (W * ff_input + b))

    return prediction

def prediction_loss(instance, prediction):
    # Compute the loss
    y_true = dy.scalarInput(1 if instance.polarity else 0)
    loss = dy.binary_log_loss(prediction, y_true)

    return loss


def build_model(missing_voc, w2v_embeddings):
    VOC_SIZE = len(missing_voc)
    WEM_DIMENSIONS = 100

    NUM_LAYERS = 1
    HIDDEN_DIM = 20

    FF_HIDDEN_DIM = 10

    print("Missing vocabulary size: %i" % len(missing_voc))

    params = dy.ParameterCollection()
    missing_wemb = params.add_lookup_parameters((VOC_SIZE, WEM_DIMENSIONS), name="missing-wemb")
    w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, init=w2v_embeddings.matrix, name="w2v-wemb")
    

    # Feed-Forward parameters
    W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM*4+1), name="W")
    b = params.add_parameters((FF_HIDDEN_DIM), name="b")
    V = params.add_parameters((1, FF_HIDDEN_DIM), name="V")

    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, params)

    ret = ModelElements(W, V, b, w2v_wemb, missing_wemb, params, builder)

    return ret

