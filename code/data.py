import pandas as pd

class DataLoader:
    """ Load datasets """

    def __init__(self, dataset_name, inpath):
        self.dataset_name = dataset_name
        self.inpath = inpath
        self.load_functions = { # mapping from dataset names to load functions
            'incels': self.load_incels,
            'white_supremacist': self.load_pandas_pickle,
            'cad': self.load_cad
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

    def load_cad(self):
        """ Load Contextual Abuse Dataset """
        self.data = pd.read_csv(self.inpath, sep='\t', index_col=0)
    
        # If only want to select hateful data
        #label_map = {
        #        'Neutral': False,
        #        'AffiliationDirectedAbuse': True,
        #        'Slur': True,
        #        'PersonDirectedAbuse': False,
        #        'IdentityDirectedAbuse': True,
        #        'CounterSpeech': False
        #    }

        #dataset.data['hate'] = dataset.data.annotation_Primary.map(label_map.get)
