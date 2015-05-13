from nltk.corpus import wordnet, stopwords
from nltk import wordpunct_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from itertools import combinations
import numpy as np

from load_data import get_train_data

sw = stopwords.words('english')

###########################
#helper methods

def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

def get_wordnet_pos(treebank_tag):
    """ translates treebank POS to wordnet (simplified) POS tags"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def prep_words(phrase):
    phrase = [w for w in phrase if w not in sw]
    return phrase

###########################
# metrics

def word_similarity(w1, w2, pos1=None, pos2=None):
    """ computes similarity between pairs of words. If pos supplied,
    will be used when performing synset look-up. Returns maximum. """
    words1 = wordnet.synsets(w1,pos1) # ONLY USING ONE WORD
    words2 = wordnet.synsets(w2,pos2) # ONLY USING ONE WORD
    if len(words1)==0 or len(words2)==0:
        return 0, 0, 0
    word1 = words1[0]
    word2 = words2[0]
    distances = []
    """
    for word1 in words1:
        for word2 in words2:
            dist = word1.wup_similarity(word2)
            if dist is not None:
                #print word1, word2, dist
                distances.append(dist)
    """
    dist = word1.wup_similarity(word2)
    if dist is not None:
        #print word1, word2, dist
        distances.append(dist)
    return distances

def sentence_similarity(entry):
    sentence = prep_words(entry.sentence_lemma)
    values = []
    for w1,w2 in combinations(sentence, 2):
            # note: using MAX similarity
            vals = word_similarity(w1,w2)
            if len(vals) > 0:
                values.append(np.power(np.nanmax(vals),2))
    return values

def phrase_similarity(entry):
    phrase  = prep_words(entry.phrase_lemma)
    context = prep_words(entry.context_lemma)
    values = []
    # below not accurate because using lemmas
    for pword in phrase:
        for sword in context:
            # note: using MAX similarity
            vals = list(word_similarity(pword,sword))
            if len(vals) > 0:
                values.append(np.power(np.nanmax(vals),2))
    return np.nanmean(values)

def mean_similarities(entry):
    vals = phrase_similarity(entry)
    ss   = sentence_similarity(entry)
    ps   = sentence_similarity(entry)
    return vals, np.nanmean(ss), np.nanmean(ps) #/np.sqrt(np.nanstd(vals))

def test():
    for line in get_train_data():
        sim = mean_similarities(line)
        print "{} ({}) - {}".format(line.phrase, line.figurative, sim)

if __name__ == "__main__":
    test()
