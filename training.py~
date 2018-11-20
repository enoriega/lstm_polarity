import csv
import random
import w2v
import itertools as it
import numpy as np
import dynet_config as dy_conf
from utils import *
from rnn import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import random

random.seed(14535)
np.random.seed(78456)
dy_conf.set(random_seed=2522620396)
python_rand_seed=65535

import dynet as dy


def main(input_path):
    with open(input_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    embeddings = w2v.load_embeddings("/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl.gz")

    print("There are %i rows" % len(data))

    instances = list([])

    instances = [Instance.from_dict(d) for d in data]
        
    # Shuffle the training instances
    random.Random(python_rand_seed).shuffle(instances)
    labels = [1 if instance.polarity else 0 for instance in instances]

    print("There are %i instances" % len(instances))

    missing_voc, missing_voc_inverse = build_vocabulary(filter(lambda w: w not in embeddings, set(it.chain.from_iterable(i.tokens for i in instances))))

    # Store the vocabulary of the missing words (from the pre-trained embeddings)
    with open("missing_vocab.txt", "w") as f:
        for i in range(len(missing_voc_inverse)):
            f.write(missing_voc_inverse[i] + "\n")

    attention_choices = {'no-att':0, '1-layer-att':1, '2-layer-att':2}
    attention_sel = attention_choices['no-att']
    elements = build_model(missing_voc, embeddings, attention_sel)

    params = elements.param_collection

    embeddings_index = WordEmbeddingIndex(elements.w2v_emb, embeddings, elements.missing_emb, missing_voc)

    # Training loop
    #trainer = dy.SimpleSGDTrainer(params, learning_rate=0.005)
    trainer = dy.AdamTrainer(params)
    trainer.set_clip_threshold(20.0)
    epochs = 100
    
    # split data and do cross-validation
    skf = StratifiedKFold(n_splits=3)
    for e in range(epochs):
        for train_indices, test_indices in skf.split(instances, labels):
            training_losses = list()
            for i, sample_index in enumerate(train_indices):
                instance = instances[sample_index]
                prediction = run_instance(instance, elements, embeddings_index, attention_sel)

                loss = prediction_loss(instance, prediction)

                loss.backward()
                trainer.update()

                loss_value = loss.value()
                training_losses.append(loss_value)

            avg_loss = np.average(training_losses)

            # Now do testing
            testing_losses = list()
            testing_predictions = list()
            testing_labels = [1 if instances[index].polarity else 0 for index in test_indices]
            for i, sample_index in enumerate(test_indices):
                instance = instances[sample_index]
                prediction = run_instance(instance, elements, embeddings_index, attention_sel)
                y_pred = 1 if prediction.value() >= 0.5 else 0
                testing_predictions.append(y_pred)
                loss = prediction_loss(instance, prediction)
                loss_value = loss.value()
                testing_losses.append(loss_value)

            f1 = f1_score(testing_labels, testing_predictions)
            precision = precision_score(testing_labels, testing_predictions)
            recall = recall_score(testing_labels, testing_predictions)

            print("Epoch %i average training loss: %f\t average testing loss: %f" % (e+1, np.average(training_losses), np.average(testing_losses)))
            print("Precision: %f\tRecall: %f\tF1: %f" % (precision, recall, f1))
            if sum(testing_predictions) >= 1:
                report = classification_report(testing_labels, testing_predictions)
                #print(report)
            if avg_loss <= 3e-3:
                break
            print()

    params.save("model.dy")


if __name__ == "__main__":
    main("SentencesInfo_all.csv")
