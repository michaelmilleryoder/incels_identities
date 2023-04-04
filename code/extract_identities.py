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


class DataLoader:
    """ Load datasets """

    def __init__(self, dataset_name, inpath):
        self.dataset_name = dataset_name
        self.inpath = inpath
        self.load_functions = { # mapping from dataset names to load functions
            'incels': self.load_incels
            'white_supremacist': self.load_pandas_pickle
        }
        self.data = None

    def load(self):
        """ Load and return dataset """
        self.load_functions[self.dataset_name]()
        return self.data

    def load_incels(self):
        self.data = pd.read_csv(self.inpath, engine='python', on_bad_lines=lambda row: row[:-2].append(' '.join(row[-2:]))) # combine last 2 elements in a line mentioning Gulag
        self.data['parsed_date'] = pd.to_datetime(self.data.date, errors='coerce') # "yesterday" etc not handled

    def load_pandas_pickle(self):
        self.data = pd.read_pickle(self.inpath)


class IdentityExtractor:
    """ Load data, identify identity term matches from a list, save out """

    def __init__(self, dataset_name, inpath, outpath, resources_paths, identity_list = 'netmapper'):
        """ Args:
                dataset_name: dataset name, to be passed to DataLoader
                inpath: path to the input data
                outpath: path to output (without ending, to be saved with JSON table pandas format and JSON lines)
                resources_paths: paths for resources used (like identity term lists)
                identity_list: str name of the identity list to use (default netmapper)
        """
        self.dataset_name = dataset_name
        self.inpath = inpath
        self.outpath = outpath
        self.resources_paths = resources_paths
        self.identity_list = identity_list
        self.identity_list_path = self.resources_paths[f'{self.identity_list}_identities']
        self.identity_exclude_path = self.resources_paths[f'{self.identity_list}_exclude']
        self.identity_include_path = self.resources_paths[f'{self.identity_list}_include']

    def load(self):
        """ Load data """
        print("Loading data...")
        data_loader = DataLoader(self.dataset_name, self.inpath)
        self.data = data_loader.load()

    def process(self, save = True):
        """ Process data and optionally save """
        self.find_identities()
        if save:
            self.save()

    def find_identities(self):
        """ Find matches in identity terms list, save matches to a column in the dataframe"""

        print("\tFinding identity matches...")

        # Load identity terms
        multi_identities = pd.read_excel(self.identity_list_path)
        with open(self.identity_exclude_path) as f:
            identity_exclude = json.load(f)
        with open(self.identity_include_path) as f:
            identity_include = json.load(f)

        # Filter to English, remove duplicates
        cols = multi_identities.columns.tolist()
        en_identities = multi_identities[cols[cols.index('English'):]].copy()
        en_identities['term'] = en_identities['English'].str.lower()
        en_identities.drop_duplicates(subset='term', inplace=True)

        # Separate out stopwords
        stops = en_identities[en_identities['stop word']==1]
        identities = en_identities[
            (en_identities['stop word']!=1) & (~en_identities['term'].isin(identity_exclude))
        ]
        self.identities = self.filter_identities(identities['term'], load_vocab=True) + identity_include

        # Search for matches
        identity_pat = re.compile(r'|'.join([(r'\b{}\b'.format(re.escape(term))) for term in self.identities]))
        zipped = list(zip(self.data.content.tolist(), itertools.repeat(identity_pat)))
        with Pool(20) as p:
            self.data[f'{self.identity_list}_identity_matches'] = p.starmap(match_identities, tqdm(zipped, ncols=80))
        # for debugging
        #self.data[f'{self.identity_list}_identity_matches'] = [match_identities(*z) for z in tqdm(zipped, ncols=80)]
        
        
    def filter_identities(self, identities, load_vocab=False):
        """ Filter identity list to only those present in the data's vocabulary, returns this list.
            Args:
                identities: list of identities to see if are in the vocab
                build_vocab: if False, will construct the vocabulary and save it to self.resources_paths['vocab_path']
                    If True, will load the vocabulary (as a JSON) from self.resources_paths['vocab_path']
        """
        # TODO: include multi-word terms that are in the data's bigrams or trigrams, too

        print("\tFiltering identity list...")

        if load_vocab:
            with open(self.resources_paths['vocab_path'], 'r') as f:
                vocab = json.load(f)
            print(f'\t\tLoaded vocab from {self.resources_paths["vocab_path"]}')

        else:
            vectorizer = CountVectorizer(ngram_range=(1,2), min_df=1)
            vectorizer.fit(self.data.content.astype(str)) # Takes ~5 min
            vocab = vectorizer.vocabulary_
            with open(self.resources_paths['vocab_path'], 'w') as f:
                json.dump(vocab, f)
            print(f'\t\tSaved vocab to {self.resources_paths["vocab_path"]}')

        present_identities = [term for term in identities if term in vocab]
        print(f'\t\t{len(present_identities)} out of {len(identities)} identity terms present in vocab')

        return present_identities

    def save(self):
        """ Save out self.data, assumed to have been processed """
        self.data.to_json(self.outpath + '.jsonl', orient='records', lines=True)
        print(f'Saved output to {self.outpath + ".jsonl"}')
        self.data.to_pickle(self.outpath + '.pkl')
        print(f'Saved output to {self.outpath + ".pkl"}')

    #def match_identities(self, text):
    #    """ Search within posts for identity matches, return them """
    #    return re.findall(self.identity_pat, str(text).lower())


def main():

    #Settings

    #dataset_name = 'incels'
    #inpath = '../../data/incels/all_comments.csv'
    #outpath = '../../data/incels/processed_comments'

    dataset_name = 'white_supremacist'
    # TODO: put this in a config file which would be passed to this file
    dataset_cfg = { 
        'inpath': '../../white_supremacist/tmp/white_supremacist_corpus.pkl',
        'outpath': '../data/white_supremacist_identities',
        'vocab_path': '../tmp/data_vocab.json',
        'text_column': 'text'
    }

    resources_paths = {
        'netmapper_identities': '../resources/generic_agents-identity_v15_2021_10_15.xlsx',
        'netmapper_exclude': '../resources/netmapper_exclude.json',
        'netmapper_include': '../resources/netmapper_include.json',
    }

    # Process the data
    extractor = IdentityExtractor(dataset_name, dataset_paths, resources_paths)
    extractor.load()
    extractor.process()


if __name__ == '__main__':
    main()
