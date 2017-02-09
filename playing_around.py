from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
import csv
import re
import string
import numpy as np

INPUT_FILEPATH = 'data/train_input.csv'
DUMP_FILEPATH = 'data/newfile.csv'
LABELS_FILEPATH = 'data/train_output.csv'
TEST_LABELS_FILEPATH = 'data/test_input.csv'
NLTK_ENGLISH_STOPWORDS = set(stopwords.words("english"))

def dump_data_to_file(filepath, data):
	with open(filepath, 'w') as dump_file:
		for line in data:
			to_write = (','.join(line) + '\n')
			print str(line), '\n'
			print to_write
			dump_file.write(to_write)

def remove_tags(string_with_tags):
	tag_regex = '<.*?>'
	return re.sub(tag_regex, '', string_with_tags)

def tokenize(sentence):
	return word_tokenize(sentence)

def nouns_only(words):
	tags = pos_tag(words)
	# not good code. Sue me.
	return [word for word, tag in tags if len(tag) >= 2 and tag[0] == 'N']

def remove_stop_words(words):
	return [word for word in words if word not in NLTK_ENGLISH_STOPWORDS]

def preprocess_and_tokenize(sentence):
	words = remove_tags(sentence)
	words = tokenize(words)
	nouns = nouns_only(words)
	print sentence
	return [word for word in remove_stop_words(nouns) if word not in string.punctuation]

def get_data(filepath, include_id = True, tokenize = True):
	data = []
	with open(filepath,'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			convo_id = [row[0]] if include_id else []
			convo = preprocess_and_tokenize(row[1]) if tokenize else [row[1]]
			#print (convo_id + convo)
			data.append(convo_id + convo)
	return data

def get_words_array(filepath):
	data = []
	with open(filepath,'rb') as csvfile:
		reader = csv.reader(csvfile)
		data = [row[1] for row in reader]
	return data[1:]

data = get_words_array(INPUT_FILEPATH)
labels = get_words_array(LABELS_FILEPATH)
cvec = CountVectorizer(analyzer='word', stop_words = 'english')
hvec = HashingVectorizer(binary = True)

'''classification = Pipeline([('vectorizer', cvec),
							#('transformer', TfidfTransformer()),
							('classifier', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])'''

classification = Pipeline([('vectorizer', cvec),
							#('transformer', TfidfTransformer()),
							('classifier', MultinomialNB())])

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0, random_state=105)

classification = classification.fit(train_data, train_labels)
#predicted = classification.predict(test_data)
#print np.mean(predicted == test_labels)
#print metrics.classification_report(test_labels, predicted)

for_submission = get_words_array(TEST_LABELS_FILEPATH)
predicted = classification.predict(for_submission)
print "id,category"
for i, line in enumerate(predicted):
	print "{},{}".format(i, line)

#print len(predicted)


