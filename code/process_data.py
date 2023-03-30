""" Load and process incels data. Includes matching identity terms.
    Saves out to JSON lines.

    @author Michael Miller Yoder
    @date 2023
"""

import re
import json
from multiprocessing import Pool
import itertools

import pandas as pd
from tqdm import tqdm


def match_identities(text, identity_pat):
    """ Search within posts for identity matches, return them """
    return re.findall(identity_pat, str(text).lower())


class DataProcessor:
    """ Load, process, and save out incels data """

    def __init__(self, inpath, outpath, resources_paths, identity_list = 'netmapper'):
        """ Args:
                inpath: path to the input data
                outpath: path to output (without ending, to be saved with JSON table pandas format and JSON lines)
                resources_paths: paths for resources used (like identity term lists)
                identity_list: str name of the identity list to use (default netmapper)
        """
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
        self.data = pd.read_csv(self.inpath, engine='python', on_bad_lines=lambda row: row[:-2].append(' '.join(row[-2:]))) # combine last 2 elements in a line mentioning Gulag
        self.data['parsed_date'] = pd.to_datetime(self.data.date, errors='coerce') # "yesterday" etc not handled

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
        #exclude = en_identities[en_identities['term'].isin(self.identity_exclude)]
        identities = en_identities[
            (en_identities['stop word']!=1) & (~en_identities['term'].isin(identity_include))
        ]
        self.identities = self.filter_identities(identities['term']) + identity_exclude

        # Search for matches
        #self.identity_pat = re.compile(r'|'.join([(r'\b{}\b'.format(re.escape(term))) for term in self.identities]))
        identity_pat = re.compile(r'|'.join([(r'\b{}\b'.format(re.escape(term))) for term in self.identities]))
        zipped = list(zip(self.data.content.tolist(), itertools.repeat(identity_pat)))
        with Pool(20) as p:
            #self.data[f'{self.identity_list}_identity_matches'] = p.starmap(self.match_identities, tqdm(self.data.content, ncols=80)) # faster but errored with TypeError: DataProcessor.match_identities() takes 2 positional arguments but 49 were given
            self.data[f'{self.identity_list}_identity_matches'] = p.starmap(match_identities, tqdm(zipped, ncols=80))

    def filter_identities(self, identities):
        """ Filter identity list to only those present in the data's vocabulary, returns this list """

        print("\tFiltering identity list...")
        pats = [re.compile(r'\b{}\b'.format(re.escape(term))) for term in identities]

        vocab = set()
        self.data.content.astype('str').str.lower().str.split().apply(vocab.update)

        identities = [term for term in identities if term in vocab]
        print(f'\t\t{len(identities)} out of {len(pats)} identity terms present in vocab')

        return identities

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
    inpath = '../../data/incels/all_comments.csv'
    outpath = '../../data/incels/processed_comments'
    resources_paths = {
        'netmapper_identities': '../resources/generic_agents-identity_v15_2021_10_15.xlsx',
        'netmapper_exclude': '../resources/netmapper_exclude.json',
        'netmapper_include': '../resources/netmapper_include.json',
    }

    # Process the data
    processor = DataProcessor(inpath, outpath, resources_paths)
    processor.load()
    processor.process()


if __name__ == '__main__':
    main()
