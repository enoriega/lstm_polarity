import csv
import re
import random
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import dynet_config as dy_conf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import collections
import pickle

random.seed(14535)
np.random.seed(78456)
dy_conf.set(random_seed=2522620396)

import dynet as dy


class Instance:

    def __init__(self, sen, start, end, trigger, polarity, rule_name):
        self.original = sen
        self.start = start  # + 1 # Plus one to account for the special start/end of sentence tokens
        self.end = end  # + 1
        self.original_trigger = trigger
        self.trigger = trigger.lower().strip()
        self.polarity = polarity  # True for positive, False for negative
        self.tokens = Instance.normalize(sen)
        self.rule_name = rule_name.lower()
        self.rule_polarity = True if self.rule_name.startswith("positive") else False;

    def get_tokens(self, k=0):
        start = max(0, self.start - k)
        end = min(len(self.tokens) - 1, self.end + k)
        return self.tokens[start:end]

    @staticmethod
    def normalize(raw):
        sentence = raw.lower()
        # Replace numbers by "[NUM]"
        sentence = re.sub(r'(\s+|^)[+-]?\d+\.?(\d+)(\s+|$)?', ' [NUM] ', sentence)
        tokens = sentence.split()

        return tokens
        # return ['[START]'] + tokens + ['[END]']

    @staticmethod
    def from_dict(d):
        return Instance(d['sentence text'],
                        int(d['event interval start']),
                        int(d['event interval end']),
                        d['trigger'],
                        # Remember the polarity is flipped because of SIGNOR
                        False if d['polarity'].startswith('Positive') else True,
                        d['rule'])

    def get_segments(self, k = 2):
        trigger_tokens = self.trigger.split()
        trigger_ix = self.tokens.index(trigger_tokens[0], self.start, self.end+1)
        tokens_prev = self.tokens[max(0, self.start - k):self.start]
        tokens_in_left = self.tokens[self.start:(trigger_ix+len(trigger_tokens)-1)]
        tokens_in_right = self.tokens[(trigger_ix+len(trigger_tokens)):self.end]
        tokens_last = self.tokens[min(self.end, len(self.tokens)-1):min(self.end+k, len(self.tokens)-1)]

        return tokens_prev, tokens_in_left, tokens_in_right, tokens_last


def build_vocabulary(data):
    index, reverse_index = dict(), dict()
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    index = dict(zip(words, range(len(words))))

    return index, reverse_index
    
def build_pubMedDict(filename):
    print('Loading Med Pub Dict ...')
    with open(filename, 'rb') as f:
        medPubDict = pickle.load(f)
    dictLen = len(medPubDict)
    dictDim = len(medPubDict['the'])
    print('Loaded Med Pub Dict !')
    return medPubDict, dictLen, dictDim

def main(input_path):
    with open(input_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print("There are %i rows" % len(data))

    instances = [Instance.from_dict(d) for d in data]

    print("There are %i instances" % len(instances))

    ix, ix_len, ix_dim = build_pubMedDict('/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl')

    VOC_SIZE = len(ix)
    WEM_DIMENSIONS = 100

    NUM_LAYERS = 1
    HIDDEN_DIM = 20

    FF_HIDDEN_DIM = 10

    print("Vocabulary size: %i" % len(ix))

    params = dy.ParameterCollection()
    #wemb = params.add_lookup_parameters((VOC_SIZE, WEM_DIMENSIONS))
    # Feed-Forward parameters
    W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1))
    b = params.add_parameters((FF_HIDDEN_DIM))
    V = params.add_parameters((1, FF_HIDDEN_DIM))
    W_a = params.add_parameters((1, HIDDEN_DIM))
    b_a = params.add_parameters(1, init=0.01)

    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, params)

    # Split training and testing
    labels = [1 if instance.polarity else 0 for instance in instances] # Compute the labels for a stratified split
    training, testing = train_test_split(instances, stratify=labels)

    print("Positive: %i\tNegative: %i" % (sum(labels), len(labels)-sum(labels)))

    testing_labels = [1 if instance.polarity else 0 for instance in testing]

    # Training loop
    trainer = dy.SimpleSGDTrainer(params, learning_rate=0.002)
    #trainer = dy.AdamTrainer(params, alpha=0.0001)
    epochs = 100
    result_save = np.zeros((epochs, 5))  # 5 columns: training loss, testing loss, recall, precision, f1.
    for e in range(epochs):
        # Shuffle the training instances
        training_losses = list()
        for i, instance in enumerate(training):

            prediction = run_instance(instance, builder, ix, W, V, b, HIDDEN_DIM, W_a, b_a)

            loss = prediction_loss(instance, prediction)

            loss.backward()
            trainer.update()

            loss_value = loss.value()
            training_losses.append(loss_value)

        avg_loss = np.average(training_losses)

        # Now do testing
        testing_losses = list()
        testing_classifications = list()
        testing_predictions = list()
        for i, instance in enumerate(testing):
            prediction = run_instance(instance, builder, ix, W, V, b, HIDDEN_DIM, W_a, b_a)
            pred_val = prediction.value()
            testing_predictions.append(pred_val)
            y_pred = 1 if prediction.value() >= 0.5 else 0
            testing_classifications.append(y_pred)
            loss = prediction_loss(instance, prediction)
            loss_value = loss.value()
            testing_losses.append(loss_value)

        f1s, precisions, recalls = dict(), dict(), dict()
        for t in np.arange(0.1, 0.9, 0.1):
            f1, precision, recall = classification_scores(testing_labels, testing_predictions, t)
            f1s[t], precisions[t], recalls[t] = f1, precision, recall

        #plot_stats(f1s, precisions, recalls, e+1)

        print("Epoch %i average training loss: %f\t average testing loss: %f" % (e+1, np.average(training_losses), np.average(testing_losses)))
        print("Precision: %f\tRecall: %f\tF1: %f" % (precisions[0.5], recalls[0.5], f1s[0.5]))
        
        result_save[e,0]=np.average(training_losses)
        result_save[e,1] = np.average(testing_losses)
        result_save[e,2] = precisions[0.5]
        result_save[e,3] = recalls[0.5]
        result_save[e,4] = f1s[0.5]
        
        if sum(testing_classifications) >= 1:
            report = classification_report(testing_labels, testing_classifications)
            print(report)
        if avg_loss <= 3e-3:
            np.savetxt('Segment_2_Attention_1L_LSTM.csv', result_save, delimiter=',')
            break
        print()
    np.savetxt('Segment_2_Attention_1L_LSTM.csv', result_save, delimiter=',')


def classification_scores(y_true, scores, threshold):
    y_pred = [1 if s >= threshold else 0 for s in scores]

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return f1, precision, recall


def plot_stats(f1s, precisions, recalls, epoch):
    x = sorted(f1s.keys())
    #plt.ion()
    plt.figure()
    plt.title("Scores for epoch %i" % epoch)
    plt.xlabel("Classification decision threshold")
    plt.ylabel("Score value")
    plt.plot(x, [f1s[k] for k in x])
    plt.plot(x, [precisions[k] for k in x])
    plt.plot(x, [recalls[k] for k in x])
    plt.legend(["F1", "Precision", "Recall"])
    plt.show()
    #plt.ioff()

def output_attention(collected_vectors, W_a, b_a):
    #print('check attention')
    collected_vectors = dy.concatenate(collected_vectors, d=1)
    a = W_a * collected_vectors + b_a
    s = dy.transpose(dy.softmax(dy.transpose(a)))
    
    return dy.reshape(s*dy.transpose(collected_vectors), d=(20,))

def run_instance(instance, builder, ix, W, V, b, HIDDEN_DIM, W_a, b_a):

    # Renew the computational graph
    dy.renew_cg()

    collected_vectors = list()

    if len(instance.get_segments())!=4:
        print("abnormal segments!")

    for segment in instance.get_segments():

        if len(segment) > 0:

            # Fetch the embeddings for the current sentence
            inputs = [dy.inputTensor(ix[w]) if w in ix else dy.inputTensor(np.zeros(100)) for w in segment]
            
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
            zero_vector = dy.zeros(HIDDEN_DIM)
            collected_vectors.append(zero_vector)

    # Concatenate the selected vectors and the polarity trigger feature
    # lstm_result = dy.concatenate(collected_vectors)
    lstm_result_att = output_attention(collected_vectors, W_a, b_a)

    trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

    ff_input = dy.concatenate([trigger_expression, lstm_result_att])

    # Run the FF network for classification
    prediction = dy.logistic(V * (W * ff_input + b))

    return prediction


def prediction_loss(instance, prediction):
    # Compute the loss
    y_true = dy.scalarInput(1 if instance.polarity else 0)
    loss = dy.binary_log_loss(prediction, y_true)

    return loss

if __name__ == "__main__":
    main("SentencesInfo_NoDuplicate.csv")
