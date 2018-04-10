import numpy as np
import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

path = '/media/alaa/Study/toxic_comment_classification/data/'

print "Load trainData..."
train = pd.read_csv(path + 'train.csv').fillna('Unknown')
print "Load testData..."
test = pd.read_csv(path + 'test.csv').fillna('Unknown')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}', # this regex matches any word and alphanumeric, excluding character 
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
print "Fit allText data to the WordVectorizer..."
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
print "Fit allText data to the CharVectorizer..."
char_vectorizer.fit(all_text)
del all_text
gc.collect()

train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
gc.collect()

del train_text
del test_text
gc.collect()
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=10, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))
print "Write Submission File..."
submission.to_csv('submission.csv', index=False)