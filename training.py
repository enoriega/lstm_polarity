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
import dynet as dy

python_rand_seed=65535
random.seed(python_rand_seed)
np.random.seed(python_rand_seed)
dy_conf.set(random_seed=python_rand_seed)





def main(input_path):
    with open(input_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    embeddings = w2v.load_embeddings("/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl.gz")

    print("There are %i rows" % len(data))

    instances = [Instance.from_dict(d) for d in data]
        
    # Shuffle the training instances
    random.Random(python_rand_seed).shuffle(instances)
    labels = [1 if instance.polarity else 0 for instance in instances]

    print("There are %i instances" % len(instances))

    attention_choices = {'no-att':0, '1-layer-att':1, '2-layer-att':2}
    attention_sel = attention_choices['no-att']
    elements = build_model(embeddings, attention_sel)

    params = elements.param_collection

    embeddings_index = WordEmbeddingIndex(elements.w2v_emb, embeddings)

    # Store the vocabulary of the missing words (from the pre-trained embeddings)
    with open("w2v_vocab.txt", "w") as f:
        for w in embeddings_index.w2v_index.to_list():
            f.write(w + "\n")

    # Training loop
    #trainer = dy.SimpleSGDTrainer(params, learning_rate=0.005)
    trainer = dy.AdamTrainer(params)

    trainer.set_clip_threshold(4.0)
    epochs = 100
    
    # split data and do cross-validation
    skf = StratifiedKFold(n_splits=5)
    for e in range(epochs):
    

        
        test_pred_dict = {}
        test_label_dict = {}
        test_loss_dict = {}
        test_reach_pred_dict={}
    
        training_losses = list()
        bad_grad_count = 0
        
        W_np = elements.W.npvalue()
        
        print('W sum:',np.sum(W_np), 'W std:',np.std(W_np))
        print('learning rate:',trainer.learning_rate)
        
        for train_indices, test_indices in skf.split(instances, labels):
            
            for i, sample_index in enumerate(train_indices):
                instance = instances[sample_index]
                prediction = run_instance(instance, elements, embeddings_index, attention_sel)

                loss = prediction_loss(instance, prediction)

                loss.backward()
                try:
                    trainer.update()
                except RuntimeError:
                    #print('encountered bad gradient, instance skipped.')
                    bad_grad_count+=1
                loss_value = loss.value()
                training_losses.append(loss_value)

            # Now do testing

            # testing_losses = list()
            # testing_predictions = list()
            # testing_labels = [1 if instances[index].polarity else 0 for index in test_indices]
            
            fold_preds = list([])
            fold_labels = list([])
            for i, sample_index in enumerate(test_indices):
                instance = instances[sample_index]
                prediction = run_instance(instance, elements, embeddings_index, attention_sel)
                y_pred = 1 if prediction.value() >= 0.5 else 0
                loss = prediction_loss(instance, prediction)
                loss_value = loss.value()
                
                if instance.neg_count not in test_pred_dict:
                    test_pred_dict[instance.neg_count]=list([])
                    test_label_dict[instance.neg_count]=list([])
                    test_loss_dict[instance.neg_count]=list([])
                    test_reach_pred_dict[instance.neg_count]=list([])
                    
                test_pred_dict[instance.neg_count].append(y_pred)
                test_label_dict[instance.neg_count].append([1 if instance.polarity else 0])
                test_loss_dict[instance.neg_count].append(loss_value)
                test_reach_pred_dict[instance.neg_count].append([1 if instance.pred_polarity else 0])
        trainer.learning_rate = trainer.learning_rate*0.1
        print('===================================================================')
        print('number of bad grads:', bad_grad_count)
        print("Epoch %i average training loss: %f" % (e+1, np.average(training_losses)))
        
        print('---------------LSTM result------------------------- ')
        all_pred = list([])
        all_label = list([])
        for neg_count in test_pred_dict.keys():
            f1 = f1_score(test_label_dict[neg_count], test_pred_dict[neg_count])
            precision = precision_score(test_label_dict[neg_count], test_pred_dict[neg_count])
            recall = recall_score(test_label_dict[neg_count], test_pred_dict[neg_count])
            print("Neg Count: %d\tN Samples: %d\tPrecision: %f\tRecall: %f\tF1: %f" % (neg_count, len(test_pred_dict[neg_count]), precision, recall, f1))
            all_pred.extend(test_pred_dict[neg_count])
            all_label.extend(test_label_dict[neg_count])
        all_f1 = f1_score(all_label, all_pred)
        print('overall f1:', all_f1)
        
        print('---------------REACH result------------------------- ')
        all_pred = list([])
        all_label = list([])
        for neg_count in test_pred_dict.keys():
            f1 = f1_score(test_label_dict[neg_count], test_reach_pred_dict[neg_count])
            precision = precision_score(test_label_dict[neg_count], test_reach_pred_dict[neg_count])
            recall = recall_score(test_label_dict[neg_count], test_reach_pred_dict[neg_count])
            print("Neg Count: %d\tN Samples: %d\tPrecision: %f\tRecall: %f\tF1: %f" % (neg_count, len(test_pred_dict[neg_count]), precision, recall, f1))
            all_pred.extend(test_reach_pred_dict[neg_count])
            all_label.extend(test_label_dict[neg_count])
        all_f1 = f1_score(all_label, all_pred)
        print('overall f1:', all_f1)
            
#            if sum(testing_predictions) >= 1:
#                report = classification_report(testing_labels, testing_predictions)
#                #print(report)
#            if avg_loss <= 3e-3:
#                break
#            print()

    #params.save("model.dy")


if __name__ == "__main__":
    main("SentencesInfo_all_label_final.csv")
