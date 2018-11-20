import re


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
        found_digit = False
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
            if re.match(r"[a-z]", c) or c == '_':
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
        trigger_ix = self.tokens.index(Instance._sanitize_word(trigger_tokens[0]), self.start, self.end+1)
        tokens_prev = self.tokens[max(0, self.start - k):self.start]
        tokens_in_left = self.tokens[self.start:(trigger_ix+len(trigger_tokens)-1)]
        tokens_in_right = self.tokens[(trigger_ix+len(trigger_tokens)):self.end]
        tokens_last = self.tokens[min(self.end, len(self.tokens)-1):min(self.end+k, len(self.tokens)-1)]

        return tokens_prev, tokens_in_left, tokens_in_right, tokens_last


class WordEmbeddingIndex(object):

    def __init__(self, w2v_data, w2v_ix, missing_data, missing_ix):
        self.w2v_data = w2v_data
        self.w2v_index = w2v_ix
        self.missing_data = missing_data
        self.missing_index = missing_ix

    def __getitem__(self, w):
        return self.w2v_data[self.w2v_index[w]] if w in self.w2v_index else self.missing_data[self.missing_index[w]]


def build_vocabulary(words):
    index, reverse_index = dict(), dict()
    for i, w in enumerate(sorted(words)):
        index[w] = i
        reverse_index[i] = w

    return index, reverse_index