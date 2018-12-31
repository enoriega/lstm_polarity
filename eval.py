import dynet as dy
from rnn import *
from utils import *
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import w2v
import csv
import random
import itertools as it
import numpy as np
import dynet_config as dy_conf

if __name__ == "__main__":
    with open("SentencesInfo_all_label.csv") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        
    instances = list([])
    for i, d in enumerate(data):
        instances.append(Instance.from_dict(d))

    p = "model.dy"
    pc = dy.ParameterCollection()

    W = pc.load_param(p, "/W")
    b = pc.load_param(p, "/b")
    V = pc.load_param(p, "/V")
    
    w_lstm_0 = pc.load_param(p, "/vanilla-lstm-builder/_0")
    w_lstm_1 = pc.load_param(p, "/vanilla-lstm-builder/_1")
    w_lstm_2 = pc.load_param(p, "/vanilla-lstm-builder/_2")
    
    missing_wemb = pc.load_lookup_param(p, "/missing-wemb")
    w2v_wemb = pc.load_lookup_param(p, "/w2v-wemb")
    
#    print(pc.parameters_list())
#    print(pc.lookup_parameters_list())
#    input('press enter to continue')
    
    embeddings = w2v.load_embeddings("/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl.gz")
    
    missing_voc, missing_voc_inverse = build_vocabulary(filter(lambda w: w not in embeddings, set(it.chain.from_iterable(i.tokens for i in instances))))
    
    embeddings_index = WordEmbeddingIndex(w2v_wemb, embeddings, missing_wemb, missing_voc)

    print('parameters loaded!')
    builder = dy.LSTMBuilder(1, 100, 30, pc)
    
#    sentence = "To formally prove that increased ROS levels enhance anti-tumour effects of the SG-free diet , the authors crossed Emu-Myc mice with mice deficient for Tigar , a fructose-2 ,6-bisphosphatase , which limits glycolysis and favours pentose phosphate pathways , thus limiting ROS levels XREF_BIBR , XREF_BIBR ( XREF_FIG ) ."
#    
#    instance = Instance.from_dict(sentence)
        
        
    test_preds = list([])
    test_labels = list([])
    for instance in instances:
        dy.renew_cg()
        collected_vectors = list()
        HIDDEN_DIM = int((W.dim()[0][1]-1)/4)
        for segment in instance.get_segments():

            if len(segment) > 0:

                # Fetch the embeddings for the current sentence
                inputs = [embeddings_index[w] for w in segment]

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
    
        y_true = dy.scalarInput(1 if instance.polarity else 0)
        loss = dy.binary_log_loss(prediction, y_true)
    
        y_pred = 1 if prediction.value() >= 0.5 else 0
        y_label = 1 if instance.polarity else 0
        
        test_preds.append(y_pred)
        test_labels.append(y_label)
        
    f1 = f1_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    
    print("Precision: %f\tRecall: %f\tF1: %f" % (precision, recall, f1))
    
