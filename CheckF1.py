import csv
import random
import w2v
import itertools as it
import numpy as np
import dynet_config as dy_conf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import sys
from collections import namedtuple

python_rand_seed = int(sys.argv[1])


print('python random seed:', python_rand_seed)



#python_rand_seed=65535
random.seed(python_rand_seed)
np.random.seed(python_rand_seed)
dy_conf.set(random_seed=python_rand_seed)

import dynet as dy
from utils import *
from rnn import run_instance, prediction_loss

ModelElements = namedtuple("ModelElements", "W V b w2v_emb c2v_embd param_collection builder_fwd builder_bwd builder_char_fwd, builder_char_bwd")

def build_model():
    params = dy.ParameterCollection()

    W,b,V, w2v_wemb, c2v_embd, builder_fwd, builder_bwd, builder_char_fwd, builder_char_bwd = dy.load("model", params)
    
    ret = ModelElements(W, V, b, w2v_wemb, c2v_embd, params, builder_fwd, builder_bwd, builder_char_fwd, builder_char_bwd)

    return ret

def main(input_path):
    with open(input_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)


    print("There are %i rows" % len(data))

    instances = [Instance.from_dict(d) for d in data]

    labels = [instance.polarity for instance in instances]
    reach_labels = [1 if instance.pred_polarity else 0 for instance in instances]

    count = 0
    print('prediction:')
    for i, label in enumerate(labels):
        print('true class:', label, '  reach class:', reach_labels[i])
        if label == reach_labels[i]:
            count+=1

    reach_f1 = f1_score(labels, reach_labels, average="micro")
    print('reach accuracy:', count/1.0/len(labels), 'reach f1:', reach_f1)

    True_0_Pred_0=0
    True_1_Pred_0=0
    True_2_Pred_0=0
    True_0_Pred_1=0
    True_1_Pred_1=0
    True_2_Pred_1=0
    True_0_Pred_2=0
    True_1_Pred_2=0
    True_2_Pred_2=0

    for i, label in enumerate(labels):
        if label==0 and reach_labels[i]==0:
            True_0_Pred_0+=1
        if label==1 and reach_labels[i]==0:
            True_1_Pred_0+=1
        if label==2 and reach_labels[i]==0:
            True_2_Pred_0+=1
        if label==0 and reach_labels[i]==1:
            True_0_Pred_1+=1
        if label==1 and reach_labels[i]==1:
            True_1_Pred_1+=1
        if label==2 and reach_labels[i]==1:
            True_2_Pred_1+=1
        if label==0 and reach_labels[i]==2:
            True_0_Pred_2+=1
        if label==1 and reach_labels[i]==2:
            True_1_Pred_2+=1
        if label==2 and reach_labels[i]==2:
            True_2_Pred_2+=1

    print('True 0 Pred 0:', True_0_Pred_0)
    print('True 1 Pred 0:', True_1_Pred_0)
    print('True 2 Pred 0:', True_2_Pred_0)
    print('True 0 Pred 1:', True_0_Pred_1)
    print('True 1 Pred 1:', True_1_Pred_1)
    print('True 2 Pred 1:', True_2_Pred_1)
    print('True 0 Pred 2:', True_0_Pred_2)
    print('True 1 Pred 2:', True_1_Pred_2)
    print('True 2 Pred 2:', True_2_Pred_2)


    # f1 = f1_score(labels, testing_predictions, average = 'samples')
    # #precision = precision_score(labels, testing_predictions)
    # #recall = recall_score(labels, testing_predictions)
    
    # #print("Precision: %f\tRecall: %f\tF1: %f" % (precision, recall, f1))
    # print("F1: %f" % (f1))

    # reach_labels = [1 if instance.pred_polarity else 0 for instance in instances]

    # reach_f1 = f1_score(reach_labels, testing_predictions, average = 'samples')
    # #precision = precision_score(labels, testing_predictions)
    # #recall = recall_score(labels, testing_predictions)
    
    # print("F1: %f" % (reach_f1))

    
    
    
if __name__ == "__main__":
    main("SentencesInfo_op_label_final_test.csv")
    
