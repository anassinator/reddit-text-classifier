from nltk import word_tokenize
from textblob import TextBlob
import csv
import re
import string

INPUT_FILEPATH = 'data/train_input.csv'
DUMP_FILEPATH = 'data/newfile.csv'
def get_data(filepath):
	data = []
	with open(filepath,'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			data.append(row)
	return data

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

def remove_punc(string_with_punc):
	return string_with_punc.translate(None, string.punctuation)

def tokenize(sentence):
	return TextBlob(sentence).words

def get_cleaned_data(filepath):
	data = []
	with open(filepath,'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			sentence = row[1]
			no_tags = remove_tags(sentence)
			words = tokenize(no_tags)
			cleaned_row = [row[0]] + words
			print cleaned_row
			data.append(cleaned_row)
	return data

data = get_cleaned_data(INPUT_FILEPATH)

