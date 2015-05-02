from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

#helper methods
def create_vector(file):
    lines = [line.strip() for line in open(file, "r")]
    return lines

train_data = create_vector("/Users/ankit/code/nlp-metaphors/data/bow/train.txt")
label_data = create_vector("/Users/ankit/code/nlp-metaphors/data/bow/label.txt")
test_data  = create_vector("/Users/ankit/code/nlp-metaphors/data/bow/test.txt")
test_label  = create_vector("/Users/ankit/code/nlp-metaphors/data/bow/test_label.txt")

assert len(train_data) == len(label_data)

# create bag of words for Training data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Training
logistic_model_data = LogisticRegression(penalty="l2")
print "Training..."
logistic_model_data.fit(X_train_tfidf, label_data)
print "Trained."

# Create bag of words for Testing data
X_new_counts = count_vect.transform(test_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

print "Testing..."
predicted = logistic_model_data.predict(X_new_tfidf)

print "---L2 Logistic Regression---"
print np.mean(predicted == test_label)
print(metrics.classification_report(test_label, predicted))


# Check svm
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, n_iter=5, random_state=42)),
                     ])

_ = text_clf.fit(train_data, label_data)
predicted = text_clf.predict(test_data)
print "--------SVM----------"
print np.mean(predicted == test_label)     
