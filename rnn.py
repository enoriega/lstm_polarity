import dynet as dy
from collections import namedtuple
import numpy as np
from utils import *


ModelElements = namedtuple("ModelElements", "W V b w2v_emb param_collection builder")

ModelElements_1 = namedtuple("ModelElements", "W V b W_a b_a w2v_emb param_collection builder")

ModelElements_2 = namedtuple("ModelElements", "W V b W_a b_a W_a_2 b_a_2 w2v_emb param_collection builder")

def output_attention(collected_vectors, W_a, b_a, HIDDEN_DIM):
    #print('check attention')
    collected_vectors = dy.concatenate(collected_vectors, d=1)
    a = W_a * collected_vectors + b_a
    s = dy.transpose(dy.softmax(dy.transpose(a)))
    
    return dy.reshape(s*dy.transpose(collected_vectors), d=(HIDDEN_DIM,))

def output_attention_low(outputs, W_a, b_a, HIDDEN_DIM):   # low-level attention (within each LSTM)
    outputs = dy.concatenate(outputs, d=1)
    a = W_a * outputs + b_a
    s = dy.transpose(dy.softmax(dy.transpose(a)))
    
    return dy.reshape(s*dy.transpose(outputs), d=(HIDDEN_DIM,))

def output_attention_high(collected_vectors, W_a_2, b_a_2, HIDDEN_DIM):   #high-level attention
    #print('check attention')
    collected_vectors = dy.concatenate(collected_vectors, d=1)
    a = W_a_2 * collected_vectors + b_a_2
    s = dy.transpose(dy.softmax(dy.transpose(a)))
    
    return dy.reshape(s*dy.transpose(collected_vectors), d=(HIDDEN_DIM,))

def run_instance(instance, model_elems, embeddings, attention_sel):

    # Renew the computational graph
    dy.renew_cg()

    builder = model_elems.builder
    builder.set_dropouts(0, 0)   # currently 0.2, 0.2 gives the best result
    
    W = model_elems.W
    V = model_elems.V
    b = model_elems.b
    collected_vectors = list()
    
    inputs = [embeddings[w] for w in instance.tokens]
    lstm = builder.initial_state()
    outputs = lstm.transduce(inputs)

    # Get the last embedding
    selected = outputs[-1]
    
    trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

    ff_input = dy.concatenate([trigger_expression, selected])

    # Run the FF network for classification
    prediction = dy.logistic(V * (W * ff_input + b))

    return prediction

def prediction_loss(instance, prediction):
    # Compute the loss
    y_true = dy.scalarInput(1 if instance.polarity else 0)
    loss = dy.binary_log_loss(prediction, y_true)

    return loss


def build_model(w2v_embeddings, attention_sel):
    WEM_DIMENSIONS = 100

    NUM_LAYERS = 1
    HIDDEN_DIM = 30

    FF_HIDDEN_DIM = 10

    params = dy.ParameterCollection()
    w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, init=w2v_embeddings.matrix, name="w2v-wemb")
    
    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, params)

    # Feed-Forward parameters
    b = params.add_parameters((FF_HIDDEN_DIM), name="b")
    V = params.add_parameters((1, FF_HIDDEN_DIM), name="V")
    
    # no attention
    if attention_sel==0:
        W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
        ret = ModelElements(W, V, b, w2v_wemb, params, builder)
        
    # 1-layer attention
    elif attention_sel==1:
        W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
        W_a = params.add_parameters((1, HIDDEN_DIM))
        b_a = params.add_parameters(1, init=0.01)
        ret = ModelElements_1(W, V, b, W_a, b_a, w2v_wemb, params, builder)
        
    # 2-layer attention
    elif attention_sel==2:
        W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
        W_a = params.add_parameters((1, HIDDEN_DIM))   #first level attention weight
        b_a = params.add_parameters(1, init=0.01)
        W_a_2 = params.add_parameters((1, HIDDEN_DIM))       # second level attention weight
        b_a_2 = params.add_parameters(1, init=0.01)
        ret = ModelElements_2(W, V, b, W_a, b_a, W_a_2, b_a_2, w2v_wemb, params, builder)

    return ret

