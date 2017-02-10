import re
import pandas as pd

def clean_data(data):
    data_size = data.shape[0]
    tag_regex = '<.*?>|\n|-|\'s?|com'
    
    cleaned_data = pd.DataFrame([ re.sub(tag_regex, '', data['conversation'][i]) \
                    for i in range(data_size) ], columns = {('conversation')})
    
    return cleaned_data

def clean_csv(input_file, output_file):
	raw_data = pd.read_csv(input_file)
	cleaned_data = clean_data(raw_data)
	cleaned_data.to_csv(output_file, index = True, header = True, index_label = 'id')

if __name__ == "__main__":
	import sys
	clean_csv((sys.argv[1]), str(sys.argv[2]))

