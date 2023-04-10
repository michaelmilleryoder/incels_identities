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

from data import Dataset


def match_identities(text, identity_pat):
    """ Search within posts for identity matches, return them """
    #all_matches = re.findall(identity_pat, str(text).lower())
    all_matches = list(re.finditer(identity_pat, str(text).lower()))
    limit = 20 # limit for number of unique identity mentions for each post
    
    res = []
    spans = []
    ctr = Counter()
    for match in all_matches:
        match_text = match.group()
        match_span = match.span()
        ctr[match_text] += 1
        if ctr[match_text] > limit:
            continue
        else:
            res.append(match_text)
            spans.append(match_span)

    # Counter method (is slightly slower)
    #ctr = Counter(all_matches)
    #res = sum([[el] * min(count, limit) for el, count in ctr.items()], [])
    return res, spans


class IdentityExtractor:
    """ Load data, identify identity term matches from a list, save out """

    def __init__(self, dataset, identities_name, identities_path,  
            identities_exclude_path=None, identities_include_path=None, load_vocab=False):
        """ Args:
                dataset: Dataset object to extract identities on
                identities_name: name of the identity list to be used
                identities_path: path to the identity list to be used
                identities_exclude_path: path to a JSON file with a list of terms in the identity list to be excluded
                identities_include_path: path to a JSON file with a list of terms to be added to the identity list
                load_vocab: If False, will extract vocab from the data and save it out to a file (specified in vocab_path).
                    If True, will load vocab from the file specified in vocab_path
        """
        self.dataset = dataset
        self.load_vocab = load_vocab
        self.identities_name = identities_name
        self.identities_path = identities_path
        self.identities_exclude_path = identities_exclude_path
        self.identities_include_path = identities_include_path
        self.identities = []

    def extract(self, save = True):
        """ Process data and optionally save """
        self.load_identities()
        self.find_identities()
        if save:
            self.dataset.save()

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
        zipped = list(zip(self.dataset.data[self.dataset.text_column].tolist(), itertools.repeat(identity_pat)))
        with Pool(20) as p:
            self.dataset.data[f'{self.identities_name}_identity_matches'], self.dataset.data[f'{self.identities_name}_identity_matches_spans'] = zip(*p.starmap(match_identities, tqdm(zipped, ncols=80)))
        # for debugging
        #self.dataset.data[f'{self.identities}_identity_matches'] = [match_identities(*z) for z in tqdm(zipped, ncols=80)]
        
        
    def filter_identities(self, identities):
        """ Filter identity list to only those present in the data's vocabulary, returns this list.
            Args:
                identities: list of identities to see if are in the vocab
        """

        print("\tFiltering identity list...")

        if self.dataset.vocab_path is None:
            self.dataset.vocab_path = f'../tmp/{self.dataset.dataset_name}_vocab.json'

        if self.load_vocab:
            with open(self.dataset.vocab_path, 'r') as f:
                vocab = json.load(f)
            print(f'\t\tLoaded vocab from {self.dataset.vocab_path}')
        else:
            print('\t\tFinding vocab...')
            vectorizer = CountVectorizer(ngram_range=(1,2), min_df=1)
            vectorizer.fit(self.dataset.dataset.data[self.dataset.dataset.text_column].astype(str)) # Takes ~5 min
            vocab = vectorizer.vocabulary_
            with open(self.vocab_path, 'w') as f:
                json.dump(vocab, f, indent=4)
            print(f'\t\tSaved vocab to {self.vocab_path}')

        present_identities = [term for term in identities if term in vocab]
        print(f'\t\t{len(present_identities)} out of {len(identities)} identity terms present in vocab')

        return present_identities

    #def match_identities(self, text):
    #    """ Search within posts for identity matches, return them """
    #    return re.findall(self.identity_pat, str(text).lower())
