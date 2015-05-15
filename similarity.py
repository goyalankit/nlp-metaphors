from nltk.corpus import wordnet, stopwords
from nltk import wordpunct_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
from itertools import combinations
import numpy as np

from load_data import (data,
                      TRAIN_FILE, TEST_FILE,
                      TEST_FILE2, TEST_FILE3)

sw = stopwords.words('english')

###########################
#helper methods

def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

def wordnet_pos(treebank_tag):
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
    words1 = np.array(wordnet.synsets(w1,pos=pos1))
    words2 = np.array(wordnet.synsets(w2,pos=pos2))
    if len(words1)==0 or len(words2)==0:
        return 0, 0, 0
    #word1 = words1[0]
    #word2 = words2[0]
    distances = []
    # NOTE: usind top 3 versions for each word
    for word1 in words1[:10]:
        for word2 in words2[:10]:
            dist = word1.wup_similarity(word2)
            if dist is not None:
                distances.append(dist)
    """
    dist = word1.wup_similarity(word2)
    if dist is not None:
        #print word1, word2, dist
        distances.append(dist)
    """
    return distances

def sentence_similarity(sentence):
    sentence = prep_words(sentence)
    values = []
    for w1,w2 in combinations(sentence, 2):
            # note: using MAX similarity
            vals = word_similarity(w1,w2)
            if len(vals) > 0:
                values.append(np.nanmax(vals))
                #values.append(np.power(np.nanmax(vals),2))
    return np.nanmean(values)

def phrase_similarity(entry):
    phrase  = prep_words(entry.phrase_lemma)
    context = prep_words(entry.context_lemma)
    phrase_pos = [(x,wordnet_pos(y).lower()) for x,y in pos_tag(phrase)]
    context_pos = [(x,wordnet_pos(y).lower()) for x,y in pos_tag(context)]

    values = []
    # below not accurate because using lemmas
    for pword,ppos in phrase_pos:
        for sword,spos in context_pos:
            vals = list(word_similarity(pword,sword,ppos,spos))
            if len(vals) > 0:
                values.append(np.nanmean(vals))
                #values.append(np.power(np.nanmax(vals),2))
    return np.nanmean(values)

def mean_similarities(entry):
    phrase_context = phrase_similarity(entry)
    #within_context = sentence_similarity(entry.context_lemma)
    #within_phrase  = sentence_similarity(entry.phrase_lemma)
    return phrase_context, phrase_context, phrase_context #within_context, within_phrase

def phrase_top_similarity(entry, bow_vals):
    pass

def get_bow(entries):
    # convert to sing string for each entry
    str_sentences = np.array([" ".join(x.sentences) for x in entries])

    # create bag of words
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    counts = count_vect.fit_transform(str_sentences)
    tfidf = tfidf_transformer.fit_transform(counts)

    labels = np.array(count_vect.get_feature_names())

    return tfidf, counts, labels


def important_similarity(entry, tfidf, labels):
    words = np.unique([w for w in entry.sentence if w not in entry.phrase])
    words = prep_words(words)
    words = np.array([w for w in words if w in labels])
    #import pdb; pdb.set_trace()
    bowi = np.array([np.where(labels==w)[0][0] for w in words])
    #import pdb; pdb.set_trace()
    bowv = tfidf[bowi]
    top3 = np.argsort(bowv)[::-1][:3]
    #import pdb; pdb.set_trace()
    dist = []
    for j in top3:
        w_bowv = bowv[j]/np.sum(bowv)
        w      = words[j]
        j_dist = []
        for pword in [x for x in entry.phrase if x not in sw]:
            #print pword, "-", w, "\t", word_similarity(w, pword)
            d = word_similarity(w, pword)
            if len(d)>0:
                j_dist.append(np.nanmean(d))
        if len(j_dist) > 0:
            dist.append(np.nanmean(j_dist)) #*w_bowv)
        else:
            dist.append(np.nan)
    dist = np.array(dist)
    try:
        return dist[~np.isnan(dist)][0] #np.nanmean(dist)
    except:
        return np.nan

def make_features(inputfile, label):
    #inputfile = TRAIN_FILE
    #label     = 'training'
    featurefile = '{}.similarity.txt'.format(inputfile)
    f = open(featurefile, 'wb')

    entries     = data(inputfile, label)
    tfidf, counts, labels  = get_bow(entries)
    for i,entry in enumerate(entries):
        w_tfidf = np.squeeze(tfidf[i,:].toarray())
        sim     = important_similarity(entry, w_tfidf, labels)
        f.write('{}\t{}\t{}\n'.format(sim,sim,sim))
        print (entry.phrase, entry.figurative, np.nanmean(sim))
    f.close()


def make_features_orig(inputfile, label):
    featurefile = '{}.similarity.txt'.format(inputfile)
    f = open(featurefile, 'wb')

    entries     = data(inputfile, label)

    for i,line in enumerate(entries):
        sim = mean_similarities(line)
        print line.phrase, sim
        f.write('{}\t{}\t{}\n'.format(sim[0],sim[1],sim[2]))
    f.close()

