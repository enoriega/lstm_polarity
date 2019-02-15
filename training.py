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

python_rand_seed = int(sys.argv[1])


print('python random seed:', python_rand_seed)



#python_rand_seed=65535
random.seed(python_rand_seed)
np.random.seed(python_rand_seed)
dy_conf.set(random_seed=python_rand_seed)

import dynet as dy
from utils import *
from rnn import *

def main(input_path):
    with open(input_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    #embeddings = w2v.load_embeddings("/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl.gz")
    embeddings = w2v.load_embeddings("/Users/zhengzhongliang/NLP_Research/2019_ASDF/medPubDict.pkl.gz")
    #embeddings = w2v.load_embeddings("/work/zhengzhongliang/ASDF_Github/2019_polarity/medPubDict.pkl.gz")


    print("There are %i rows" % len(data))

    instances = [Instance.from_dict(d) for d in data]

    char_embeddings = build_char_dict(instances)

    # Shuffle the training instances
    random.Random(python_rand_seed).shuffle(instances)
    labels = [1 if instance.polarity else 0 for instance in instances]

    print("There are %i instances" % len(instances))
    
    training, testing = train_test_split(instances, stratify=labels)

    print("Positive: %i\tNegative: %i" % (sum(labels), len(labels)-sum(labels)))

    testing_labels = [1 if instance.polarity else 0 for instance in testing]

    # char_embd_choices = {'no-char-embd':0, 'biGRU-char-embd':1}
    # char_embd_sel = char_embd_choices['biGRU-char-embd']
    # word_embd_choices = {'no-med-pub':0,'med-pub':1}
    # word_embd_sel = word_embd_choices['no-med-pub']


    
    # Store the vocabulary of the missing words (from the pre-trained embeddings)
#    with open("w2v_vocab.txt", "w") as f:
#        for w in embeddings_index.w2v_index.to_list():
#            f.write(w + "\n")

    # Training loop
    #trainer = dy.SimpleSGDTrainer(params, learning_rate=0.005)

    # use this to test whether a smaller learning rate can boost the performance of pre-trained models. delete
    # this line when generating formal results.
    # trainer.learning_rate = trainer.learning_rate*0.5
    
    # split data and do cross-validation
    element = build_model(embeddings, char_embeddings)
    embeddings_index = WordEmbeddingIndex(element.w2v_emb, embeddings)
    embeddings_char_index = CharEmbeddingIndex(element.c2v_embd, char_embeddings)
    param = element.param_collection
    trainer = dy.AdamTrainer(param)
    trainer.set_clip_threshold(4.0)
    
    epochs=5
    for e in range(epochs):
        # Shuffle the training instances
        training_losses = list()
        for i, instance in enumerate(training):

            #prediction = run_instance(instance.get_tokens(), instance.rule_polarity, elements, embeddings_index)
            prediction = run_instance(instance, element, embeddings_index, embeddings_char_index)

            loss = prediction_loss(instance, prediction)

            loss.backward()
            trainer.update()

            loss_value = loss.value()
            training_losses.append(loss_value)

        avg_loss = np.average(training_losses)

        # Now do testing
        testing_losses = list()
        testing_predictions = list()
        for i, instance in enumerate(testing):
            #prediction = run_instance(instance.get_tokens(), instance.rule_polarity, elements, embeddings_index)
            prediction = run_instance(instance, element, embeddings_index, embeddings_char_index)

            y_pred = 1 if prediction.value() >= 0.5 else 0
            testing_predictions.append(y_pred)
            loss = prediction_loss(instance, prediction)
            loss_value = loss.value()
            testing_losses.append(loss_value)
            
        trainer.learning_rate = trainer.learning_rate*0.1

        f1 = f1_score(testing_labels, testing_predictions)
        precision = precision_score(testing_labels, testing_predictions)
        recall = recall_score(testing_labels, testing_predictions)

        print("Epoch %i average training loss: %f\t average testing loss: %f" % (e+1, np.average(training_losses), np.average(testing_losses)))
        print("Precision: %f\tRecall: %f\tF1: %f" % (precision, recall, f1))
        if sum(testing_predictions) >= 1:
            report = classification_report(testing_labels, testing_predictions)
            print(report)
        if avg_loss <= 3e-3:
            break
            
    print("wirting to disk...")
    dy.save("model",[element.W, element.b, element.V, element.w2v_emb, element.c2v_embd, element.builder_fwd, element.builder_bwd, element.builder_char_fwd, element.builder_char_bwd])
    print("finished writing to disk!")


if __name__ == "__main__":
    main("SentencesInfo_all_label_final_train.csv")
