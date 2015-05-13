import re
import os
import pickle
import numpy as np
from nltk import wordpunct_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from pattern.en import parse
from pattern.search import search

lemmatize = WordNetLemmatizer().lemmatize

TRAIN_FILE = "data/subtask5b_en_allwords_train.txt"
TEST_FILE  = "data/subtask5b_en_allwords_test.txt"
TEST_FILE2 = "data/subtask5b_en_lexsample_test.txt"
TEST_FILE3 = "data/subtask5b_en_lexsample_train.txt"

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

class Entry(object):
    figurative = False
    sentence   = None
    phrase     = None
    pos        = None
    lemma      = None
    context    = None
    def __init__(self):
        pass

def load_data(data_file):
    corpus_data = []
    corpus_target = []
    print "Reading data file: {}".format(data_file)
    corpus_file = open (data_file, "r")

    print "Importing data..."
    lines = []
    for line in corpus_file:
        entry = Entry()
        line_parts = line.split("\t")
        # data validity check
        assert len(line_parts) == 4
        entry.figurative = True if (line_parts[2] == "figuratively") else False
        # initial pre-process
        phrase     = line_parts[1].decode('utf8').lower()
        sentences  = remove_tags(line_parts[3].decode('utf8').lower())

        entry.phrase       = wordpunct_tokenize(phrase)
        entry.phrase_lemma = [lemmatize(w) for w in entry.phrase]

        # clean up and parse sentence
        entry.sentences = sent_tokenize(sentences)
        entry.sentence  = np.array([wordpunct_tokenize(x) for x in entry.sentences])
        #entry.pos       = pos_tag(entry.sentence)
        entry.sentence  = np.hstack(entry.sentence)
        entry.sentence_lemma = np.array([lemmatize(w) for w in entry.sentence])

        # find match of phrase (original strings)
        phrase_match = search(" ".join(entry.phrase_lemma),
                              " ".join(entry.sentence_lemma))
        if len(phrase_match) > 0:
            # isolate context (remove phrase)
            context_select = np.ones(len(entry.sentence), dtype=np.bool)
            start  = phrase_match[0].start
            stop   = phrase_match[0].stop
            context_select[start:stop] = False
            entry.context       = entry.sentence[context_select]
            entry.context_lemma = entry.sentence_lemma[context_select]
        else:
            #print u"phrase {} not found in sentence {}?".format(phrase, sentences)
            entry.context = entry.sentence
            entry.context_lemma = entry.sentence_lemma

        lines.append(entry)
    return lines

def data(filename, label, force_reload=False):
    pklfile = '{}.data'.format(label)
    if os.path.exists(pklfile):
        entries = pickle.load( open(pklfile, 'rb'))
    else:
        entries = load_data(filename)
        pickle.dump(entries, open(pklfile, "wb"))
    return entries

