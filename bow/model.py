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


def train(features, labels):
    # Training
    logistic_model_data = LogisticRegression(penalty="l2")
    print "Training..."
    logistic_model_data.fit(combined_train_featues, label_data)
    print "Trained."
    return logistic_model_data

def test(model, features):
    print "Testing..."
    predicted = model.predict(combined_test_features)
    print "Tested."
    return predicted


def get_stats(predicted, test_label):
    print "---L2 Logistic Regression---"
    print np.mean(predicted == test_label)
    print(metrics.classification_report(test_label, predicted))
    print "----------------------------"


# ***********
# Script Start
# ***********

# Set LEX to True to run the model on unseen data
LEX = True

# Set USE_SRL to True to run the model with Verb Restrictions SRL feature
USE_SRL = False

if LEX == True:
    train_data = create_vector("../data/bow/lex_train.txt")
    label_data = create_vector("../data/bow/lex_train_label.txt")
    test_data  = create_vector("../data/bow/lex_test.txt")
    test_label  = create_vector("../data/bow/lex_test_label.txt")
else:
    train_data = create_vector("../data/bow/train.txt")
    label_data = create_vector("../data/bow/label.txt")
    test_data  = create_vector("../data/bow/test.txt")
    test_label  = create_vector("../data/bow/test_label.txt")

# SRL features
if USE_SRL:
    if LEX == True:
        srl_train_features = create_feature_vec("../data/semverb/features_train_vec_lex.txt")
        srl_test_features = create_feature_vec("../data/semverb/features_test_vec_lex.txt")
    else:
        srl_train_features = create_feature_vec("../data/semverb/features_train_vec.txt")
        srl_test_features = create_feature_vec("../data/semverb/features_test_vec.txt")
else:
    pass

assert len(train_data) == len(label_data)

# create bag of words for Training data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape

# Create bag of words for Testing data
X_new_counts = count_vect.transform(test_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
print X_new_tfidf.shape


# TRAINING: Merge Bag of words and SRL features
if USE_SRL:
    combined_train_featues = hstack([X_train_tfidf, srl_train_features])
else:
    combined_train_featues = X_train_tfidf

# TESTING Merge Bag of words and SRL features
if USE_SRL:
    combined_test_features = hstack([X_new_tfidf, srl_test_features])
else:
    combined_test_features = X_new_tfidf

logistic_model = train(combined_train_featues, label_data)
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

