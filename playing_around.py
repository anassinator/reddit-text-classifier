from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from textblob import TextBlob
import csv
import re
import string

INPUT_FILEPATH = 'data/train_input.csv'
DUMP_FILEPATH = 'data/newfile.csv'
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

def get_cleaned_words(sentence):
	words = remove_tags(sentence)
	words = tokenize(words)
	nouns = nouns_only(words)
	return [word for word in remove_stop_words(nouns) if word not in string.punctuation]

def get_cleaned_data(filepath):
	data = []
	with open(filepath,'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			sentence = row[1]
			cleaned_words = get_cleaned_words(sentence)
			cleaned_row = [row[0]] + cleaned_words
			print cleaned_row
			data.append(cleaned_row)
	return data

data = get_cleaned_data(INPUT_FILEPATH)
dump_data_to_file(DUMP_FILEPATH, data)

