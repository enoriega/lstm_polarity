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

    embeddings = w2v.load_embeddings("/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl.gz")
    #embeddings = w2v.load_embeddings("/Users/zhengzhongliang/NLP_Research/2019_ASDF/medPubDict.pkl.gz")
    #embeddings = w2v.load_embeddings("/work/zhengzhongliang/ASDF_Github/2019_polarity/medPubDict.pkl.gz")


    print("There are %i rows" % len(data))

    instances = [Instance.from_dict(d) for d in data]

    char_embeddings = build_char_dict(instances)

    # Shuffle the training instances
    random.Random(python_rand_seed).shuffle(instances)
    labels = [1 if instance.polarity else 0 for instance in instances]
    
    element = build_model()
    embeddings_index = WordEmbeddingIndex(element.w2v_emb, embeddings)
    embeddings_char_index = CharEmbeddingIndex(element.c2v_embd, char_embeddings)
    param = element.param_collection
    
    testing_losses = list()
    testing_predictions = list()
    for i, instance in enumerate(instances):
        #prediction = run_instance(instance.get_tokens(), instance.rule_polarity, elements, embeddings_index)
        prediction = run_instance(instance, element, embeddings_index, embeddings_char_index)

        y_pred = 1 if prediction.value() >= 0.5 else 0
        testing_predictions.append(y_pred)
        loss = prediction_loss(instance, prediction)
        loss_value = loss.value()
        testing_losses.append(loss_value)

    f1 = f1_score(labels, testing_predictions)
    precision = precision_score(labels, testing_predictions)
    recall = recall_score(labels, testing_predictions)
    
    print("Precision: %f\tRecall: %f\tF1: %f" % (precision, recall, f1))
    
if __name__ == "__main__":
    main("SentencesInfo_all_label_final.csv")
    
