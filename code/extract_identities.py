""" Matching identity term lists in text datasets.
    Saves out to JSON lines with a column including extracted identity term mentions.

    @author Michael Miller Yoder
    @date 2023
"""

import re
import json
from multiprocessing import Pool
import itertools
import pdb
from collections import Counter

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from data import DataLoader


def match_identities(text, identity_pat):
    """ Search within posts for identity matches, return them """
    all_matches = re.findall(identity_pat, str(text).lower())
    limit = 20 # limit for number of unique identity mentions for each post
    
    res = []
    ctr = Counter()
    for match in all_matches:
        ctr[match] += 1
        if ctr[match] > limit:
            continue
        else:
            res.append(match)

    # Counter method (is slightly slower)
    #ctr = Counter(all_matches)
    #res = sum([[el] * min(count, limit) for el, count in ctr.items()], [])
    return res


class IdentityExtractor:
    """ Load data, identify identity term matches from a list, save out """

    def __init__(self, dataset_name, inpath, outpath, identities_name, identities_path,  
            load_vocab=False, vocab_path=None, text_column='text', tokenize=False, 
            identities_exclude_path=None, identities_include_path=None):
        """ Args:
                dataset_name: dataset name, to be passed to DataLoader
                inpath: path to the input data to extract identities from
                outpath: path to output (without ending, to be saved with JSON table pandas format and JSON lines)
                identities_name: name of the identity list to be used
                identities_path: path to the identity list to be used
                load_vocab: If False, will extract vocab from the data and save it out to a file (specified in vocab_path).
                    If True, will load vocab from the file specified in vocab_path
                vocab_path: path to a JSON file with the extracted vocabulary from the data (keys ngrams, values counts).
                    If None, will default to ../tmp/{dataset_name}_vocab.json
                text_column: name of the column in the input data that contains the text
                tokenize: whether to tokenize the input column. If this is done, the original content will be saved in <text_column>_orig
                identities_exclude_path: path to a JSON file with a list of terms in the identity list to be excluded
                identities_include_path: path to a JSON file with a list of terms to be added to the identity list
        """
        self.dataset_name = dataset_name
        self.inpath = inpath
        self.outpath = outpath
        self.load_vocab = load_vocab
        if vocab_path is None:
            self.vocab_path = f'../tmp/{self.dataset_name}_vocab.json'
        else:
            self.vocab_path = vocab_path 
        self.text_column = text_column
        self.tokenize = tokenize

        self.identities_name = identities_name
        self.identities_path = identities_path
        self.identities_exclude_path = identities_exclude_path
        self.identities_include_path = identities_include_path
        self.identities = []

    def load(self):
        """ Load data """
        print("Loading data...")
        data_loader = DataLoader(self.dataset_name, self.inpath, self.tokenize, self.text_column)
        self.data = data_loader.load()

    def extract(self, save = True):
        """ Process data and optionally save """
        self.load_identities()
        self.find_identities()
        if save:
            self.save()

    def load_identities(self):
        """ Load list of identities to extract """
        print("Loading identity term list...")

        if self.identities_name == 'netmapper':

            # Load identity terms
            multi_identities = pd.read_excel(self.identities_path)
            with open(self.identities_exclude_path) as f:
                identities_exclude = json.load(f)
            with open(self.identities_include_path) as f:
                identities_include = json.load(f)

            # Filter to English, remove duplicates
            cols = multi_identities.columns.tolist()
            en_identities = multi_identities[cols[cols.index('English'):]].copy()
            en_identities['term'] = en_identities['English'].str.lower()
            en_identities.drop_duplicates(subset='term', inplace=True)

            # Separate out stopwords
            identities = en_identities[
                (en_identities['stop word']!=1) & (~en_identities['term'].isin(identities_exclude))
            ]
            self.identities = self.filter_identities(identities['term'].tolist() + identities_include)


    def find_identities(self):
        """ Find matches in identity terms list, save matches to a column in the dataframe"""

        print("\tFinding identity matches...")

        # Search for matches
        identity_pat = re.compile(r'|'.join([(r'\b{}\b'.format(re.escape(term))) for term in self.identities]))
        zipped = list(zip(self.data[self.text_column].tolist(), itertools.repeat(identity_pat)))
        with Pool(20) as p:
            self.data[f'{self.identities_name}_identity_matches'] = p.starmap(match_identities, tqdm(zipped, ncols=80))
        # for debugging
        #self.data[f'{self.identities}_identity_matches'] = [match_identities(*z) for z in tqdm(zipped, ncols=80)]
        
        
    def filter_identities(self, identities):
        """ Filter identity list to only those present in the data's vocabulary, returns this list.
            Args:
                identities: list of identities to see if are in the vocab
        """

        print("\tFiltering identity list...")

        if self.load_vocab:
            with open(self.vocab_path, 'r') as f:
                vocab = json.load(f)
            print(f'\t\tLoaded vocab from {self.vocab_path}')

        else:
            print('\t\tFinding vocab...')
            vectorizer = CountVectorizer(ngram_range=(1,2), min_df=1)
            vectorizer.fit(self.data[self.text_column].astype(str)) # Takes ~5 min
            vocab = vectorizer.vocabulary_
            with open(self.vocab_path, 'w') as f:
                json.dump(vocab, f, indent=4)
            print(f'\t\tSaved vocab to {self.vocab_path}')

        present_identities = [term for term in identities if term in vocab]
        print(f'\t\t{len(present_identities)} out of {len(identities)} identity terms present in vocab')

        return present_identities

    def save(self):
        """ Save out self.data, assumed to have been processed """
        print("Saving...")
        self.data.to_pickle(self.outpath + '.pkl')
        print(f'Saved output to {self.outpath + ".pkl"}')
        self.data.to_json(self.outpath + '.jsonl', orient='records', lines=True)
        print(f'Saved output to {self.outpath + ".jsonl"}')

    #def match_identities(self, text):
    #    """ Search within posts for identity matches, return them """
    #    return re.findall(self.identity_pat, str(text).lower())
