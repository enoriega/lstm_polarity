import csv
import re
import random
import w2v
import itertools as it
import numpy as np
import dynet_config as dy_conf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

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
    def _is_number(w):
        i = 0
        found_digit = True
        while i < len(w):
            c = w[i]
            match = re.match("\r", c)
            if match is None and c != '-' and c != '+' and c != ',' and c != '.' and c != '/' and c != '\\':
                return False
            if match:
                found_digit = True
            i += 1

        return found_digit

    @staticmethod
    def _sanitize_word(uw, keepNumbers=True):
        w = uw.lower()

        # skip parens from corenlp
        if w == "-lrb-" or w == "-rrb-" or w == "-lsb-" or w == "-rsb-":
            return ""

        # skip URLS
        if w.startswith("http") or ".com" in w or ".org" in w:  # added .com and .org to cover more urls (becky)
            return ""

        # normalize numbers to a unique token
        if Instance._is_number(w):
            return "xnumx" if keepNumbers else ""

        # remove all non-letters; convert letters to lowercase
        os = ""
        i = 0
        while i < len(w):
            c = w[i]
            # added underscore since it is our delimiter for dependency stuff...
            if re.match(r"[a-z]", w) or c == '_':
                os += c
            i += 1

        return os

    @staticmethod
    def normalize(raw):
        sentence = raw.lower()
        # Replace numbers by "[NUM]"
        #sentence = re.sub(r'(\s+|^)[+-]?\d+\.?(\d+)(\s+|$)?', ' [NUM] ', sentence)
        tokens = [Instance._sanitize_word(t) for t in sentence.split()]

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

    def get_segments(self, k=2):
        trigger_tokens = self.trigger.split()
        trigger_ix = self.tokens.index(trigger_tokens[0], self.start, self.end+1)
        tokens_prev = self.tokens[max(0, self.start - k):self.start]
        tokens_in_left = self.tokens[self.start:(trigger_ix+len(trigger_tokens)-1)]
        tokens_in_right = self.tokens[(trigger_ix+len(trigger_tokens)):self.end]
        tokens_last = self.tokens[min(self.end, len(self.tokens)-1):min(self.end+k, len(self.tokens)-1)]

        return tokens_prev, tokens_in_left, tokens_in_right, tokens_last


def build_vocabulary(words):
    index, reverse_index = dict(), dict()
    for i, w in enumerate(sorted(words)):
        index[w] = i
        reverse_index[i] = w

    return index, reverse_index


def main(input_path):
    with open(input_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    embeddings = w2v.load_embeddings("medPubDict.pkl.gz")

    print("There are %i rows" % len(data))

    instances = [Instance.from_dict(d) for d in data]

    print("There are %i instances" % len(instances))

    ix, rix = build_vocabulary(filter(lambda w: w not in embeddings, set(it.chain.from_iterable(i.tokens for i in instances))))

    # Store the vocabulary
    with open("vocab.txt", "w") as f:
        for i in range(len(rix)):
            f.write(rix[i] + "\n")


    VOC_SIZE = len(ix)
    WEM_DIMENSIONS = 100

    NUM_LAYERS = 1
    HIDDEN_DIM = 20

    FF_HIDDEN_DIM = 10

    print("Vocabulary size: %i" % len(ix))

    params = dy.ParameterCollection()
    missing_wemb = params.add_lookup_parameters((VOC_SIZE, WEM_DIMENSIONS), name="missing-wemb")
    w2v_wemb = params.add_lookup_parameters(embeddings.matrix.shape, init=embeddings.matrix, name="w2v-wemb")

    # Feed-Forward parameters
    W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM), name="W")
    b = params.add_parameters((FF_HIDDEN_DIM), name = "b")
    V = params.add_parameters((1, FF_HIDDEN_DIM), name = "V")

    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, params)

    # Split training and testing
    labels = [1 if instance.polarity else 0 for instance in instances] # Compute the labels for a stratified split
    training, testing = train_test_split(instances, stratify=labels)

    print("Positive: %i\tNegative: %i" % (sum(labels), len(labels)-sum(labels)))

    testing_labels = [1 if instance.polarity else 0 for instance in testing]

    # Training loop
    trainer = dy.SimpleSGDTrainer(params)
    epochs = 100
    for e in range(epochs):
        # Shuffle the training instances
        training_losses = list()
        for i, instance in enumerate(training):

            prediction = run_instance(instance, builder, missing_wemb, w2v_wemb, embeddings, ix, W, V, b)

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
            prediction = run_instance(instance, builder, missing_wemb, w2v_wemb, embeddings, ix, W, V, b)
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
            print(report)
        if avg_loss <= 3e-3:
            break
        print()

    params.save("model.dy")


def run_instance(instance, builder, missing_wemb, w2v_wemb, embeddings, ix, W, V, b):

    # Renew the computational graph
    dy.renew_cg()

    # Fetch the embeddings for the current sentence
    words = instance.get_tokens()
    inputs = [w2v_wemb[embeddings[w]] if w in embeddings else missing_wemb[ix[w]] for w in words]

    # Run FF over the LSTM
    lstm = builder.initial_state()
    outputs = lstm.transduce(inputs)

    # Get the last embedding
    selected = outputs[-1]

    # Run the FF network for classification
    prediction = dy.logistic(V * (W * selected + b))

    return prediction


def prediction_loss(instance, prediction):
    # Compute the loss
    y_true = dy.scalarInput(1 if instance.polarity else 0)
    loss = dy.binary_log_loss(prediction, y_true)

    return loss

if __name__ == "__main__":
    main("SentencesInfo.csv")
