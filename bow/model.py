from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from scipy.sparse import coo_matrix, hstack
from stop_words import STOP_WORDS
import random
import pickle

PRINT_LEVEL = ''#'VERBOSE'
USE_DEV = True

def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

def create_feature_vec(file):
    data = create_vector(file)
    data_array = np.array(data, dtype='float64')
    return data_array.reshape(len(data_array), 1)

def convert_to_feature_vec(data):
    data_array = np.array(data, dtype='float64')
    data_array[np.where(np.isnan(data_array))] = 0
    return data_array.reshape(len(data_array), 1)


def train(features, labels):
    # Training
    logistic_model_data = LogisticRegression(penalty="l2")
    if PRINT_LEVEL == 'VERBOSE': print "Training..."
    logistic_model_data.fit(features, labels)
    if PRINT_LEVEL == 'VERBOSE': print "Trained."
    return logistic_model_data

def test(model, features):
    if PRINT_LEVEL == 'VERBOSE': print "Testing..."
    predicted = model.predict(features)
    if PRINT_LEVEL == 'VERBOSE': print "Tested."
    return predicted


def get_stats(predicted, test_label):
    if PRINT_LEVEL == 'VERBOSE': print "---L2 Logistic Regression---"
    accuracy = np.mean(predicted == test_label)
    if PRINT_LEVEL == 'VERBOSE': print accuracy
    if PRINT_LEVEL == 'VERBOSE': print(metrics.classification_report(test_label, predicted))
    if PRINT_LEVEL == 'VERBOSE': print "----------------------------"
    return accuracy

def get_similarity_vec(file):
    data_similarity_vec = create_vector(file)
    number = 0
    phrase_sentence = []
    only_phrase     = []
    only_sentence   = []
    for line in data_similarity_vec:
        #import pdb; pdb.set_trace()
        sline = line.split("\t")
        phrase_sentence.append(sline[0])
        only_sentence.append(sline[1])
        only_phrase.append(sline[2])

    phrase_sentence_features = convert_to_feature_vec(phrase_sentence)
    only_sentence_features   = convert_to_feature_vec(only_sentence)
    only_phrase_features     = convert_to_feature_vec(only_phrase)
    return phrase_sentence_features, only_sentence_features, only_phrase_features


def model(stop_word_list=None):
    # ***********
    # Script Start
    # ***********

    if USE_BOW:
        if LEX == True:
            train_data = create_vector("../data/bow/lex_train.txt")
            if USE_DEV:
                test_data  = create_vector("../data/bow/lex_dev.txt")
            else:
                test_data  = create_vector("../data/bow/lex_test.txt")
        else:
            train_data = create_vector("../data/bow/train.txt")
            if USE_DEV:
                test_data  = create_vector("../data/bow/allwords_dev.txt")
            else:
                test_data  = create_vector("../data/bow/test.txt")

        # create bag of words for Training data
        if stop_word_list == "default":
            count_vect = CountVectorizer(stop_words='english')
        elif stop_word_list is None:
            count_vect = CountVectorizer()
        else:
            count_vect = CountVectorizer(stop_words=stop_word_list)


        X_train_counts = count_vect.fit_transform(train_data)

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        # Create bag of words for Testing data
        X_new_counts = count_vect.transform(test_data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    else:
        pass

    if LEX == True:
        label_data = create_vector("../data/bow/lex_train_label.txt")
        if USE_DEV:
            test_label  = create_vector("../data/bow/lex_dev_label.txt")
        else:
            test_label  = create_vector("../data/bow/lex_test_label.txt")
    else:
        label_data = create_vector("../data/bow/label.txt")
        if USE_DEV:
            test_label  = create_vector("../data/bow/allwords_dev_label.txt")
        else:
            test_label  = create_vector("../data/bow/test_label.txt")

    # SRL features
    if USE_SRL:
        if PRINT_LEVEL == 'VERBOSE': print "Using SRL Feature"
        if LEX == True:
            srl_train_features = create_feature_vec("../data/semverb/features_train_vec_lex.txt")
            if USE_DEV:
                srl_test_features = create_feature_vec("../data/semverb/features_dev_vec_lex.txt")
            else:
                srl_test_features = create_feature_vec("../data/semverb/features_test_vec_lex.txt")
        else:
            srl_train_features = create_feature_vec("../data/semverb/features_train_vec.txt")
            if USE_DEV:
                srl_test_features = create_feature_vec("../data/semverb/features_dev_vec.txt")
            else:
                srl_test_features = create_feature_vec("../data/semverb/features_test_vec.txt")
    else:
        pass

    if SIMILARITY_FEATURES:
        if LEX:
            #phrase_sentence_test_features, only_sentence_test_features, only_phrase_test_features = \
            #        get_similarity_vec("../data/sem/subtask5b_en_lexsample_test.txt.similarity.txt")
            #phrase_sentence_train_features, only_sentence_train_features, only_phrase_train_features = \
            #        get_similarity_vec("../data/sem/subtask5b_en_lexsample_train.txt.similarity.txt")

            print "MARK NOT USING YOUR FEATURES. UNCOMMENT TO USE YOUR FEATURE"
            phrase_sentence_test_features, only_sentence_test_features, only_phrase_test_features = \
                    get_similarity_vec("../data/other-data/test_features.txt")
            phrase_sentence_train_features, only_sentence_train_features, only_phrase_train_features = \
                    get_similarity_vec("../data/other-data/train_features.txt")
        else:
            phrase_sentence_test_features, only_sentence_test_features, only_phrase_test_features = \
                    get_similarity_vec("../data/sem/subtask5b_en_allwords_test.txt.similarity.txt")
            phrase_sentence_train_features, only_sentence_train_features, only_phrase_train_features = \
                    get_similarity_vec("../data/sem/subtask5b_en_allwords_train.txt.similarity.txt")

    combined_train_features = None
    combined_test_features = None

    if USE_BOW:
        combined_train_features = X_train_tfidf
        combined_test_features = X_new_tfidf
        if PRINT_LEVEL == 'VERBOSE': print "------ Feature: BOW --------\n"
        if PRINT_LEVEL == 'VERBOSE': print "Train shape: %s" % str(combined_train_features.shape)
        if PRINT_LEVEL == 'VERBOSE': print "Test shape: %s" % str(combined_test_features.shape)

    # TRAINING: Merge Bag of words and SRL features
    if USE_SRL:
        if PRINT_LEVEL == 'VERBOSE': print "Using SRL"
        if combined_train_features is not None:
            combined_train_features = hstack([combined_train_features, srl_train_features])
            combined_test_features = hstack([combined_test_features, srl_test_features])
        else:
            combined_train_features = srl_train_features
            combined_test_features = srl_test_features

        if PRINT_LEVEL == 'VERBOSE': print "------ Feature: SRL --------\n"
        if PRINT_LEVEL == 'VERBOSE': print "Train shape: %s" % str(combined_train_features.shape)
        if PRINT_LEVEL == 'VERBOSE': print "Test shape: %s" % str(combined_test_features.shape)

    if USE_PHRASE_SENTENCE:
        if combined_train_features is not None:
            combined_train_features = hstack([combined_train_features, phrase_sentence_train_features])
            combined_test_features  = hstack([combined_test_features, phrase_sentence_test_features])
        else:
            combined_test_features  = phrase_sentence_test_features
            combined_train_features = phrase_sentence_train_features

        if PRINT_LEVEL == 'VERBOSE': print "------ Feature: USE_PHRASE_SENTENCE --------\n"
        if PRINT_LEVEL == 'VERBOSE': print "Train shape: %s" % str(combined_train_features.shape)
        if PRINT_LEVEL == 'VERBOSE': print "Test shape: %s" % str(combined_test_features.shape)

    if USE_ONLY_SENTENCE:
        if combined_train_features is not None:
            combined_train_features = hstack([combined_train_features, only_sentence_train_features])
            combined_test_features = hstack([combined_test_features, only_sentence_test_features])
        else:
            combined_test_features = only_sentence_test_features
            combined_train_features = only_sentence_train_features

        if PRINT_LEVEL == 'VERBOSE': print "------ Feature: USE_ONLY_SENTENCE --------\n"
        if PRINT_LEVEL == 'VERBOSE': print "Train shape: %s" % str(combined_train_features.shape)
        if PRINT_LEVEL == 'VERBOSE': print "Test shape: %s" % str(combined_test_features.shape)

    if USE_ONLY_PHRASE:
        if combined_train_features is not None:
            combined_train_features = hstack([combined_train_features, only_phrase_train_features])
            combined_test_features = hstack([combined_test_features, only_phrase_test_features])
        else:
            combined_test_features = only_phrase_test_features
            combined_train_features = only_phrase_train_features

        if PRINT_LEVEL == 'VERBOSE': print "------ Feature: USE_ONLY_PHRASE --------\n"
        if PRINT_LEVEL == 'VERBOSE': print "Train shape: %s" % str(combined_train_features.shape)
        if PRINT_LEVEL == 'VERBOSE': print "Test shape: %s" % str(combined_test_features.shape)

    if PRINT_LEVEL == 'VERBOSE': print "----------------------------"
    if PRINT_LEVEL == 'VERBOSE': print "--- Running the model ---"

    logistic_model = train(combined_train_features, label_data)
    predicted      = test(logistic_model, combined_test_features)
    return get_stats(predicted, test_label)


    """
    print "Printing advanced statistics"
    test_vector = create_vector("../data/subtask5b_en_lexsample_test.txt")
    current_phrase = None
    previous_phrase = None
    current_count = 0
    num_phrases = 0
    previous_count = 0
    total_sen_count, feature_count = combined_test_features.shape
    for phrase in test_vector:
        current_phrase = phrase.split('\t')[1]
        if previous_phrase is None:
            previous_phrase = current_phrase

        if current_phrase != previous_phrase:
            print ("Running for '%s' with current phrase being. %s Total number of previous phrases: " % (str(previous_phrase), str(current_phrase))), (current_count - previous_count)
            combined_test_features_t = combined_test_features.toarray()[previous_count:current_count-1, ]
            p_t = test(logistic_model, combined_test_features_t)
            get_stats(p_t, test_label[previous_count:current_count-1])
            previous_count = current_count
            previous_phrase = current_phrase

        current_count += 1
    """

def get_random_stopwords():
    slist = []
    for word in STOP_WORDS:
        if random.random() > 0.5:
            slist.append(word)
        else:
            pass
    print "List size: ", len(slist)
    return slist


def increment_counts(improves_hash, rlist):
    for word in rlist:
        improves_hash[word] += 1

# Set LEX to True to run the model on seen data
LEX = False

USE_BOW = True

# Set USE_SRL to True to run the model with Verb Restrictions SRL feature
USE_SRL = True

# You must set similarity features to True if you are using any of the relatedness feature
# This doesn't work with developement data
SIMILARITY_FEATURES = False
USE_PHRASE_SENTENCE = False
USE_ONLY_SENTENCE = False
USE_ONLY_PHRASE = False

class Parameters(object):
    word_list  = []
    seen_accuracy = 0.0
    unseen_accuracy = 0.0

    def __init__(self, wl, aseen, aunseen):
        word_list = wl
        seen_accuracy = aseen
        unseen_accuracy = aunseen

improves_seen = {}
improves_unseen = {}
improves_both = {}
list_that_improves_both = {}
for word in STOP_WORDS:
    improves_seen[word] = 0
    improves_unseen[word] = 0
    improves_both[word] = 0


BASE_LINE_UNSEEN = 0.6198
BASE_LINE_SEEN = 0.8016

parameter_objects = []

for i in range(0,50):
    print "iteration count #: ", i
    rlist = get_random_stopwords()
    LEX = False
    unseen_accuracy = model(rlist)
    print unseen_accuracy
    LEX = True
    seen_accuracy = model(rlist)
    print seen_accuracy

    if seen_accuracy > BASE_LINE_SEEN:
        increment_counts(improves_seen, rlist)

    if unseen_accuracy > BASE_LINE_SEEN:
        increment_counts(improves_unseen, rlist)

    if ((seen_accuracy > BASE_LINE_SEEN) & (unseen_accuracy > BASE_LINE_UNSEEN)):
        print '----------- both improved start-----------\n'
        LEX = False
        USE_DEV = False
        PRINT_LEVEL = 'VERBOSE'

        print '------ unseen ----'
        model(rlist)
        LEX = True
        print '------ seen ----'
        model(rlist)
        increment_counts(improves_both, rlist)
        pobj = Parameters(rlist, seen_accuracy, unseen_accuracy)
        pobj.word_list = rlist
        pobj.seen_accuracy = seen_accuracy
        pobj.unseen_accuracy = unseen_accuracy
        parameter_objects.append(pobj)
        print "--------- both improved end ------------\n"
        USE_DEV = True
        PRINT_LEVEL = 'DEFAULT'

with open('seen_improvement.pickle', 'wb') as f:
    pickle.dump(improves_seen, f, pickle.HIGHEST_PROTOCOL)

with open('unseen_improvement.pickle', 'wb') as f:
    pickle.dump(improves_unseen, f, pickle.HIGHEST_PROTOCOL)

with open('both_improvement.pickle', 'wb') as f:
    pickle.dump(improves_both, f, pickle.HIGHEST_PROTOCOL)

with open('both_improvement_object.pickle', 'wb') as f:
    pickle.dump(parameter_objects, f, pickle.HIGHEST_PROTOCOL)
# Check svm
#text_clf = Pipeline([('vect', CountVectorizer()),
                     #('tfidf', TfidfTransformer()),
                     #('clf', SGDClassifier(loss='hinge', penalty='l2',
                         #alpha=1e-3, n_iter=5, random_state=42)),
                     #])

#_ = text_clf.fit(train_data, label_data)
#predicted = text_clf.predict(test_data)
#print "--------SVM----------"
#print np.mean(predicted == test_label)     


