""" Load and process incels data. Includes matching identity terms.
    Saves out to JSON lines.

    @author Michael Miller Yoder
    @date 2023
"""

import re
from multiprocessing import Pool
import itertools

import pandas as pd
from tqdm import tqdm


def match_identities(text, identity_pat):
    """ Search within posts for identity matches, return them """
    return re.findall(identity_pat, str(text).lower())


class DataProcessor:
    """ Load, process, and save out incels data """

    def __init__(self, inpath, outpath, resources_paths, identity_list = 'netmapper', identity_exclude = [], identity_include=[]):
        """ Args:
                inpath: path to the input data
                outpath: path to output (without ending, to be saved with JSON table pandas format and JSON lines)
                resources_paths: paths for resources used (like identity term lists)
                identity_list: str name of the identity list to use (default netmapper)
                identity_exclude: list of terms to exclude from the identity list
                identity_exclude: list of terms to add to the identity list
        """
        self.inpath = inpath
        self.outpath = outpath
        self.resources_paths = resources_paths
        self.identity_list = identity_list
        self.identity_list_path = self.resources_paths[f'{self.identity_list}_identities']
        self.identity_exclude = identity_exclude
        self.identity_include = identity_include

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

        # Filter to English, remove duplicates
        cols = multi_identities.columns.tolist()
        en_identities = multi_identities[cols[cols.index('English'):]].copy()
        en_identities['term'] = en_identities['English'].str.lower()
        en_identities.drop_duplicates(subset='term', inplace=True)

        # Separate out stopwords
        stops = en_identities[en_identities['stop word']==1]
        #exclude = en_identities[en_identities['term'].isin(self.identity_exclude)]
        identities = en_identities[
            (en_identities['stop word']!=1) & (~en_identities['term'].isin(self.identity_exclude))
        ]
        self.identities = self.filter_identities(identities['term']) + self.identity_include

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
        'netmapper_identities': '../resources/generic_agents-identity_v15_2021_10_15.xlsx'
    }
    identity_exclude = ['don', 'other', 'others', 'friend', 'friends', 'people', 'who', 'asshole', 'dick',
               'character', 'person', 'people', 'majority', 'bot', 'everyone', 'everyone here',
                'officially', 'tech', 'individual', 'worker', 'workers', 'giant', 'human', 'humans', 'ass',
                'nobody', 'brother', 'sister', 'mother', "mother's", 'father', 'daughter', 'son', 'mom', 'wife', 'wives', 'husband', 'husbands', 'cousin', 'cousins',
                'they all', 'count', 'god', 'general', 'user', 'users', 'member', 'members', 'english', 'finish', 'slayer', 'speaker',
                'misogynist', 'king', 'queen', 'rn', 'fellow', 'buddy', 'enemies', 'corpse', 'revolutionary', 'gymnast', 'messiah', 'jesus', 'embryo',
                'dr', 'doctor', 'dahmer', 'characters', 'cheat', 'sexist', 'professional', 'client', 'mate', 'dad', 'customers', 'assholes', 'whose',
                'mama', 'co-workers', 'employees', 'uncle', 'hermit', 'ogre', 'potter', 'phantom', 'dwellers', 'saviour', 'prophet', 'morons', 'guide',
                'majors', 'partners', 'villain', 'agent', 'model', 'juggernaut', 'ego', 'avatar', 'player', 'dragon', 'pm', 'winner', 'winners', 'surrogate', 'nudes',
            'blogger', 'bloggers', 'loser', 'losers', 'adult', 'avatars', 'partner', 'idiot', 'clown', 'bully', 'slave', 'adults',
        'hunter', 'troll', 'hunters', 'trolls', 'master', 'masters', 'victim', 'driver', 'slaves',
               ]
    identity_include = [
        'jewish', 'kikes', 'judaism', 'goy', 'goyim', 'black people', 'white people',
        'black woman', 'black man', 'black women', 'black men', 'white women', 'white woman', 'white man', 'white men',
        'non white', 'non whites', 'white trash',
        'gay', 'trans', 'transgender', 'bi', 'bisexual', 'bisexuals', 'pansexual', 'pansexuals', 'non-binary', 'genderqueer',
        'homo', 'transman', 'transwoman', 'queers', 'fags',
    ]

    # Process the data
    processor = DataProcessor(inpath, outpath, resources_paths, 
        identity_exclude=identity_exclude, identity_include=identity_include)
    processor.load()
    processor.process()


if __name__ == '__main__':
    main()
