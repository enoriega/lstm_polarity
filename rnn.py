import dynet as dy
from collections import namedtuple
import numpy as np
from utils import *


ModelElements = namedtuple("ModelElements", "W V b w2v_emb c2v_embd param_collection builder builder_char_fwd  builder_char_bwd")

ModelElements_1 = namedtuple("ModelElements", "W V b W_a b_a w2v_emb c2v_embd param_collection builder builder_char_fwd  builder_char_bwd")

ModelElements_2 = namedtuple("ModelElements", "W V b W_a b_a W_a_2 b_a_2 w2v_emb c2v_embd param_collection builder builder_char_fwd  builder_char_bwd")

def get_char_embd(word, model_elems, embeddings_char_index):
    gru_char_fwd = model_elems.builder_char_fwd.initial_state()
    gru_char_bwd = model_elems.builder_char_bwd.initial_state()

    #print('current word:', word)
    #print('number of characters in word:',len(word))

    if word=='':
        char_embd_list = [embeddings_char_index['']]

    else:
        char_embd_list = list([])
        for character in word:
            char_embd_list.append(embeddings_char_index[character])

    output_fwd = gru_char_fwd.transduce(char_embd_list)
    output_bwd = gru_char_bwd.transduce(char_embd_list[::-1])

    # print('length of input:', len(output_fwd))
    # print('length of output:', len(output_bwd))

    char_embd_vec = dy.concatenate([output_fwd[-1], output_bwd[-1]],d=0)

    # print('char embd vec dim:', char_embd_vec.dim())

    # input('press enter to continue')

    return char_embd_vec

def output_attention(collected_vectors, W_a, b_a, HIDDEN_DIM):
    #print('check attention')
    collected_vectors = dy.concatenate(collected_vectors, d=1)
    a = W_a * collected_vectors + b_a
    s = dy.transpose(dy.softmax(dy.transpose(a)))

    #print('attention score:',s.npvalue())
    
    return dy.reshape(s*dy.transpose(collected_vectors), d=(HIDDEN_DIM,))

def output_attention_low(outputs, W_a, b_a, HIDDEN_DIM):   # low-level attention (within each LSTM)
    outputs = dy.concatenate(outputs, d=1)
    a = W_a * outputs + b_a
    s = dy.transpose(dy.softmax(dy.transpose(a)))

    #print('attention score:',s.npvalue())

    
    return dy.reshape(s*dy.transpose(outputs), d=(HIDDEN_DIM,))

def output_attention_high(collected_vectors, W_a_2, b_a_2, HIDDEN_DIM):   #high-level attention
    #print('check attention')
    collected_vectors = dy.concatenate(collected_vectors, d=1)
    a = W_a_2 * collected_vectors + b_a_2
    s = dy.transpose(dy.softmax(dy.transpose(a)))

    #print('attention score:',s.npvalue())

    
    return dy.reshape(s*dy.transpose(collected_vectors), d=(HIDDEN_DIM,))

def run_instance(instance, model_elems, embeddings, char_embeddings, seg_sel, att_sel):

    # Renew the computational graph
    dy.renew_cg()

    builder = model_elems.builder
    builder.set_dropouts(0, 0)   # currently 0.2, 0.2 gives the best result
    
    W = model_elems.W
    V = model_elems.V
    b = model_elems.b
    collected_vectors = list()
    

    if att_sel==0:
        # 1-segment using tokens
        if seg_sel==0:
            HIDDEN_DIM = int((W.dim()[0][1]-1))
            collected_vectors = list()

            # it is ok to have empty embedding in the token sequence. 
            inputs = list([])
            for word in instance.tokens:
                word_embd = embeddings[word]
                char_embd = get_char_embd(word, model_elems, char_embeddings)
                input_vec = dy.concatenate([word_embd,char_embd], d=0)
                inputs.append(input_vec)

            
            #inputs = [embeddings[w] for w in instance.tokens] # in Enrique's master branch code, he uses get_toekn
            lstm = builder.initial_state()
            outputs = lstm.transduce(inputs)

            # Get the last embedding
            selected = outputs[-1]
            
            trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

            ff_input = dy.concatenate([trigger_expression, selected])

            # Run the FF network for classification
            prediction = dy.logistic(V * (W * ff_input + b))

            return prediction

        # 1-segment using get_tokens
        elif seg_sel==1:
            HIDDEN_DIM = int((W.dim()[0][1]-1))
            collected_vectors = list()

            # it is ok to have empty embedding in the token sequence. 
            inputs = list([])
            for word in instance.get_tokens():
                word_embd = embeddings[word]
                char_embd = get_char_embd(word, model_elems, char_embeddings)
                input_vec = dy.concatenate([word_embd,char_embd], d=0)
                inputs.append(input_vec)

            
            #inputs = [embeddings[w] for w in instance.tokens] # in Enrique's master branch code, he uses get_toekn
            lstm = builder.initial_state()
            outputs = lstm.transduce(inputs)

            # Get the last embedding
            selected = outputs[-1]
            
            trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

            ff_input = dy.concatenate([trigger_expression, selected])

            # Run the FF network for classification
            prediction = dy.logistic(V * (W * ff_input + b))

            return prediction
            
        # four segments
        elif seg_sel==2:
            HIDDEN_DIM = int((W.dim()[0][1]-1)/4)
            for segment in instance.get_segments():

                if len(segment) > 0:

                    # Fetch the embeddings for the current sentence
                    inputs = list([])
                    for word in segment:
                        word_embd = embeddings[word]
                        char_embd = get_char_embd(word, model_elems, char_embeddings)
                        input_vec = dy.concatenate([word_embd,char_embd], d=0)
                        inputs.append(input_vec)

                    #inputs = [embeddings[w] for w in segment]

                    # Run FF over the LSTM
                    lstm = builder.initial_state()
                    outputs = lstm.transduce(inputs)

                    # Get the last embedding
                    selected = outputs[-1]

                    # Collect it
                    collected_vectors.append(selected)
                else:
                    zero_vector = dy.zeros(HIDDEN_DIM)
                    collected_vectors.append(zero_vector)

            # Concatenate the selected vectors and the polarity trigger feature
            lstm_result = dy.concatenate(collected_vectors)  # shape=4*20

            trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

            ff_input = dy.concatenate([trigger_expression, lstm_result])

            # Run the FF network for classification
            prediction = dy.logistic(V * (W * ff_input + b))

        
    # 1-layer attention
    elif att_sel==1:
        if seg_sel==0:
            HIDDEN_DIM = int((W.dim()[0][1]-1))

            W_a = model_elems.W_a
            b_a = model_elems.b_a

            collected_vectors = list()

            # it is ok to have empty embedding in the token sequence. 
            inputs = list([])
            for word in instance.tokens:
                word_embd = embeddings[word]
                char_embd = get_char_embd(word, model_elems, char_embeddings)
                input_vec = dy.concatenate([word_embd,char_embd], d=0)
                inputs.append(input_vec)
            
            #inputs = [embeddings[w] for w in instance.tokens] # in Enrique's master branch code, he uses get_toekn
            lstm = builder.initial_state()
            outputs = lstm.transduce(inputs)

            # Get the last embedding
            selected = output_attention(outputs, W_a, b_a, HIDDEN_DIM)
            
            trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

            ff_input = dy.concatenate([trigger_expression, selected])

            # Run the FF network for classification
            prediction = dy.logistic(V * (W * ff_input + b))

        elif seg_sel==1:
            HIDDEN_DIM = int((W.dim()[0][1]-1))

            W_a = model_elems.W_a
            b_a = model_elems.b_a

            collected_vectors = list()

            # it is ok to have empty embedding in the token sequence. 
            inputs = list([])
            for word in instance.get_tokens():
                word_embd = embeddings[word]
                char_embd = get_char_embd(word, model_elems, char_embeddings)
                input_vec = dy.concatenate([word_embd,char_embd], d=0)
                inputs.append(input_vec)
            
            #inputs = [embeddings[w] for w in instance.tokens] # in Enrique's master branch code, he uses get_toekn
            lstm = builder.initial_state()
            outputs = lstm.transduce(inputs)

            # Get the last embedding
            selected = output_attention(outputs, W_a, b_a, HIDDEN_DIM)
            
            trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

            ff_input = dy.concatenate([trigger_expression, selected])

            # Run the FF network for classification
            prediction = dy.logistic(V * (W * ff_input + b))


        elif seg_sel==2:
            HIDDEN_DIM = int((W.dim()[0][1])-1)
        
            W_a = model_elems.W_a
            b_a = model_elems.b_a

            for segment in instance.get_segments():
                if len(segment) > 0:
                    inputs = list([])
                    for word in segment:
                        word_embd = embeddings[word]
                        char_embd = get_char_embd(word, model_elems, char_embeddings)
                        input_vec = dy.concatenate([word_embd,char_embd], d=0)
                        inputs.append(input_vec)
                    #inputs = [embeddings[w] for w in segment]
                    lstm = builder.initial_state()
                    outputs = lstm.transduce(inputs)
                    selected = outputs[-1]
                    collected_vectors.append(selected)
                else:
                    zero_vector = dy.zeros(HIDDEN_DIM)
                    collected_vectors.append(zero_vector)

            lstm_result_att = output_attention(collected_vectors, W_a, b_a, HIDDEN_DIM)
            trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)
            ff_input = dy.concatenate([trigger_expression, lstm_result_att])
            prediction = dy.logistic(V * (W * ff_input + b))
        
    elif att_sel==2:
        HIDDEN_DIM = int((W.dim()[0][1])-1)
    
        W_a = model_elems.W_a
        b_a = model_elems.b_a
        W_a_2 = model_elems.W_a_2
        b_a_2 = model_elems.b_a_2
        
        for segment in instance.get_segments():
            if len(segment) > 0:
                inputs = list([])
                for word in segment:
                    word_embd = embeddings[word]
                    char_embd = get_char_embd(word, model_elems, char_embeddings)
                    input_vec = dy.concatenate([word_embd,char_embd], d=0)
                    inputs.append(input_vec)
                #inputs = [embeddings[w] for w in segment]
                lstm = builder.initial_state()
                outputs = lstm.transduce(inputs)
                selected_att = output_attention_low(outputs, W_a, b_a, HIDDEN_DIM)
                collected_vectors.append(selected_att)
            else:
                zero_vector = dy.zeros(HIDDEN_DIM)
                collected_vectors.append(zero_vector)

        lstm_result_att = output_attention_high(collected_vectors, W_a_2, b_a_2, HIDDEN_DIM)
        trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)
        ff_input = dy.concatenate([trigger_expression, lstm_result_att])
        prediction = dy.logistic(V * (W * ff_input + b))

    #input('press enter to continue')
    return prediction

def prediction_loss(instance, prediction):
    # Compute the loss
    y_true = dy.scalarInput(1 if instance.polarity else 0)
    loss = dy.binary_log_loss(prediction, y_true)

    return loss


def build_model(w2v_embeddings, char_embeddings, seg_sel, att_sel):
    WEM_DIMENSIONS = 100

    NUM_LAYERS = 1
    HIDDEN_DIM = 30

    FF_HIDDEN_DIM = 10
    CEM_DIMENSIONS = 20


    params = dy.ParameterCollection()
    #w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, init=w2v_embeddings.matrix, name="w2v-wemb")
    w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, name="w2v-wemb")
    c2v_embd = params.add_lookup_parameters((len(char_embeddings)+1, CEM_DIMENSIONS), name="c2v-emb")


    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS+CEM_DIMENSIONS*2, HIDDEN_DIM, params)
    builder_char_fwd = dy.GRUBuilder(NUM_LAYERS, CEM_DIMENSIONS, CEM_DIMENSIONS, params)
    builder_char_bwd = dy.GRUBuilder(NUM_LAYERS, CEM_DIMENSIONS, CEM_DIMENSIONS, params)

    # Feed-Forward parameters
    b = params.add_parameters((FF_HIDDEN_DIM), name="b")
    V = params.add_parameters((1, FF_HIDDEN_DIM), name="V")
    
    # no attention
    if att_sel==0:
        if seg_sel==0 or seg_sel==1:
            W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
        elif seg_sel==2:
            W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM*4+1), name="W")
        ret = ModelElements(W, V, b, w2v_wemb, c2v_embd, params, builder, builder_char_fwd, builder_char_bwd)
        
    # 1-layer attention
    elif att_sel==1:

        W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
        W_a = params.add_parameters((1, HIDDEN_DIM))
        b_a = params.add_parameters(1, init=0.01)
        ret = ModelElements_1(W, V, b, W_a, b_a, w2v_wemb, c2v_embd, params, builder, builder_char_fwd, builder_char_bwd)
        
    # 2-layer attention
    elif att_sel==2:
        W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
        W_a = params.add_parameters((1, HIDDEN_DIM))   #first level attention weight
        b_a = params.add_parameters(1, init=0.01)
        W_a_2 = params.add_parameters((1, HIDDEN_DIM))       # second level attention weight
        b_a_2 = params.add_parameters(1, init=0.01)
        ret = ModelElements_2(W, V, b, W_a, b_a, W_a_2, b_a_2, w2v_wemb, c2v_embd, params, builder, builder_char_fwd, builder_char_bwd)

    return ret
