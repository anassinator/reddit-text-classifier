import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def tokenize(data):
	stop = set(stopwords.words('english'))
	tokenizer = RegexpTokenizer(r'\w+')
	
	tokenized_data = [tokenizer.tokenize(' '.join([word for word in convo.split() if word not in stop])) \
			  for convo in data['conversation']]
	return tokenized_data

def clean_data(data):
	data_size = data.shape[0]
	tag_regex = '<.*?>|\n|-|\'s?|com'

	cleaned_data = pd.DataFrame([ re.sub(tag_regex, '', data['conversation'][i]) \
				    for i in range(data_size) ], columns = {('conversation')})
	return cleaned_data

def clean_data_pipeline(data):
	data = tokenize(clean_data(data))
	return data

def clean_csv(input_file, output_file, tokenize = False):
	raw_data = pd.read_csv(input_file)

	if tokenize : cleaned_data = pd.DataFrame(clean_data_pipeline(raw_data))
	else :        cleaned_data = clean_data(raw_data)

	cleaned_data.to_csv(output_file, header = False, index = True)

if __name__ == "__main__":
	import sys, argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("ifile", help="input file path")
	parser.add_argument("ofile", help="output file path")
	parser.add_argument("-t", "--tokenize", help="option to tokenize the data", action="store_true")
	args = parser.parse_args()

	clean_csv(args.ifile, args.ofile, args.tokenize)

