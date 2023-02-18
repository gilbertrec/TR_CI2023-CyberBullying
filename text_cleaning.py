import pandas as pd
import re

paths = ['aggression_parsed_dataset', 'attack_parsed_dataset', 'kaggle_parsed_dataset', 'toxicity_parsed_dataset', 'twitter_parsed_dataset', 'youtube_parsed_dataset']

def clean_text(data):
    url_regex = 'http\S+'
    annotation_regex = '@[A-Za-z0-9_]+'
    punctuation_regex = '[^\w\s]'
    youtube_regex = '(watchfeature|watchv)\S+'   

    data['Text'] = data['Text'].apply(lambda text: re.sub(r'({})|({})|({})|({})'.format(url_regex, annotation_regex, punctuation_regex, youtube_regex), '', str(text)))
    data['Text'] = data['Text'].apply(lambda text: str(text).replace(u'xa0', u' ').strip())
    data = data[data['Text'].str.len() >= 1]
    data.drop_duplicates(subset=['Text'])
    
    return data


for path in paths:
    data = pd.read_csv('./dataset/{}.csv'.format(path))
    data = clean_text(data)
    data.to_csv('./dataset/{}.csv'.format(path), index=False)
    
    print(data.head())
    print('Dataset size:', data.shape)
    print('Columns are:', data.columns)


