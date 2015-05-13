from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from scipy.sparse import coo_matrix, hstack

#helper methods
def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

def create_feature_vec(file):
    data = create_vector(file)
    data_array = np.array(data, dtype='float64')
    return data_array.reshape(len(data_array), 1)

def convert_to_feature_vec(data):
    data_array = np.array(data, dtype='float64')
    return data_array.reshape(len(data_array), 1)


def train(features, labels):
    # Training
    logistic_model_data = LogisticRegression(penalty="l2")
    print "Training..."
    logistic_model_data.fit(features, label_data)
    print "Trained."
    return logistic_model_data

def test(model, features):
    print "Testing..."
    predicted = model.predict(features)
    print "Tested."
    return predicted


def get_stats(predicted, test_label):
    print "---L2 Logistic Regression---"
    print np.mean(predicted == test_label)
    print(metrics.classification_report(test_label, predicted))
    print "----------------------------"

def get_similarity_vec(file):
    data_similarity_vec = create_vector(file)
    number = 0
    phrase_sentence = []
    only_phrase     = []
    only_sentence   = []
    for line in data_similarity_vec:
        sline = line.split("\t")
        phrase_sentence.append(sline[0])
        only_sentence.append(sline[1])
        only_phrase.append(sline[2])

    phrase_sentence_features = convert_to_feature_vec(phrase_sentence)
    only_sentence_features   = convert_to_feature_vec(only_sentence)
    only_phrase_features     = convert_to_feature_vec(only_phrase)
    return phrase_sentence_features, only_sentence_features, only_phrase_features


# ***********
# Script Start
# ***********

# Set LEX to True to run the model on unseen data
LEX = False

USE_BOW = True

# Set USE_SRL to True to run the model with Verb Restrictions SRL feature
USE_SRL = False

# You must set similarity features to True if you are using any of the relatedness feature
SIMILARITY_FEATURES = False
USE_PHRASE_SENTENCE = False
USE_ONLY_SENTENCE = False
USE_ONLY_PHRASE = False

if USE_BOW:
    if LEX == True:
        train_data = create_vector("../data/bow/lex_train.txt")
        test_data  = create_vector("../data/bow/lex_test.txt")
    else:
        train_data = create_vector("../data/bow/train.txt")
        test_data  = create_vector("../data/bow/test.txt")

    # create bag of words for Training data
    count_vect = CountVectorizer()
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
    test_label  = create_vector("../data/bow/lex_test_label.txt")
else:
    label_data = create_vector("../data/bow/label.txt")
    test_label  = create_vector("../data/bow/test_label.txt")

# SRL features
if USE_SRL:
    print "Using SRL Feature"
    if LEX == True:
        srl_train_features = create_feature_vec("../data/semverb/features_train_vec_lex.txt")
        srl_test_features = create_feature_vec("../data/semverb/features_test_vec_lex.txt")
    else:
        srl_train_features = create_feature_vec("../data/semverb/features_train_vec.txt")
        srl_test_features = create_feature_vec("../data/semverb/features_test_vec.txt")
else:
    pass

if SIMILARITY_FEATURES:
    if LEX:
        phrase_sentence_test_features, only_sentence_test_features, only_phrase_test_features = get_similarity_vec("../data/subtask5b_en_lexsample_test.txt.similarity.txt")
        phrase_sentence_train_features, only_sentence_train_features, only_phrase_train_features = get_similarity_vec("../data/subtask5b_en_lexsample_train.txt.similarity.txt")
    else:
        phrase_sentence_test_features, only_sentence_test_features, only_phrase_test_features = get_similarity_vec("../data/subtask5b_en_allwords_test.txt.similarity.txt")
        phrase_sentence_train_features, only_sentence_train_features, only_phrase_train_features = get_similarity_vec("../data/subtask5b_en_allwords_train.txt.similarity.txt")

combined_train_features = None
combined_test_features = None

if USE_BOW:
    combined_train_features = X_train_tfidf
    combined_test_features = X_new_tfidf
    print "------ Feature: BOW --------\n"
    print "Train shape: %s" % str(combined_train_features.shape)
    print "Test shape: %s" % str(combined_test_features.shape)

# TRAINING: Merge Bag of words and SRL features
if USE_SRL:
    print "Using SRL"
    if combined_train_features is not None:
        combined_train_features = hstack([combined_train_features, srl_train_features])
        combined_test_features = hstack([combined_test_features, srl_test_features])
    else:
        combined_train_features = srl_train_features
        combined_test_features = srl_test_features

    print "------ Feature: SRL --------\n"
    print "Train shape: %s" % str(combined_train_features.shape)
    print "Test shape: %s" % str(combined_test_features.shape)

if USE_PHRASE_SENTENCE:
    if combined_train_features is not None:
        combined_train_featrues = hstack([combined_train_features, phrase_sentence_train_features])
        combined_test_features = hstack([combined_test_features, phrase_sentence_test_features])
    else:
        combined_test_features = phrase_sentence_test_features
        combined_train_features = phrase_sentence_train_features

    print "------ Feature: USE_PHRASE_SENTENCE --------\n"
    print "Train shape: %s" % str(combined_train_features.shape)
    print "Test shape: %s" % str(combined_test_features.shape)

if USE_ONLY_SENTENCE:
    if combined_train_features is not None:
        combined_train_featrues = hstack([combined_train_features, only_sentence_train_features])
        combined_test_features = hstack([combined_test_features, only_sentence_test_features])
    else:
        combined_test_features = only_sentence_test_features
        combined_train_features = only_sentence_train_features

    print "------ Feature: USE_ONLY_SENTENCE --------\n"
    print "Train shape: %s" % str(combined_train_features.shape)
    print "Test shape: %s" % str(combined_test_features.shape)

if USE_ONLY_PHRASE:
    if combined_train_features is not None:
        combined_train_featrues = hstack([combined_train_features, only_phrase_train_features])
        combined_test_features = hstack([combined_test_features, only_phrase_test_features])
    else:
        combined_test_features = only_phrase_test_features
        combined_train_features = only_phrase_train_features

    print "------ Feature: USE_ONLY_PHRASE --------\n"
    print "Train shape: %s" % str(combined_train_features.shape)
    print "Test shape: %s" % str(combined_test_features.shape)

print "----------------------------"
print "--- Running the model ---"

logistic_model = train(combined_train_features, label_data)
predicted      = test(logistic_model, combined_test_features)
get_stats(predicted, test_label)

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

