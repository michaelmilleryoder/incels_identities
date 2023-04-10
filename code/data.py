import pandas as pd
import spacy
from tqdm import tqdm


class Dataset:
    """ Load datasets, tokenize and lowercase if need. Stores metadata with the data, too """

    def __init__(self, dataset_name, inpath, outpath=None, tokenize=False, text_column=None, vocab_path=None):
        """ 
            Args:
                dataset_name: dataset name
                inpath: input path to the data
                outpath: path to output, optional, to save the data to if processing is done
                     (without ending, to be saved with JSON table pandas format and JSON lines)
                    If None, will default to ../tmp/{dataset_name}_vocab.json
                tokenize: whether to tokenize the input column. If this is done, the original content will be saved in <text_column>_orig
                text_column: (only if tokenizing) the name of the column of text.
                    After tokenizing, the original content will be kept in <text_column>_orig
                vocab_path: path to a JSON file with the extracted vocabulary from the data (keys ngrams, values counts). 
                    Optional, default None
        """
        self.dataset_name = dataset_name
        self.inpath = inpath
        self.outpath = outpath
        self.tokenize = tokenize
        if self.tokenize:
            self.nlp = spacy.load('en_core_web_sm', exclude=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        self.text_column = text_column
        self.vocab_path = vocab_path
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
        if self.tokenize:
            print("Tokenizing data...")
            self.data = pd.read_csv(self.inpath, engine='python', on_bad_lines=lambda row: row[:-2].append(' '.join(row[-2:]))) # combine last 2 elements in a line mentioning Gulag
            self.data['parsed_date'] = pd.to_datetime(self.data.date, errors='coerce') # "yesterday" etc not handled
            self.data[f'{self.text_column}_orig'] = self.data[self.text_column]
            # TODO: speed up with Pool and nlp.tokenizer(text), or try nlp.tokenizer.pipe(texts)
            #self.data[self.text_column] = [' '.join([tok.text.lower() for tok in doc]) for doc in self.nlp.pipe(
            #        tqdm(self.data[self.text_column].astype(str), ncols=80))]
            self.data[self.text_column] = [' '.join([tok.text.lower() for tok in doc]) for doc in self.nlp.pipe(
                    tqdm(self.data[self.text_column].astype(str), ncols=80))]
        else:
            self.load_pandas_pickle()

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

    def save(self):
        """ Save out self.data, assumed to have been processed """
        print("Saving...")
        self.data.to_pickle(self.outpath + '.pkl')
        print(f'Saved output to {self.outpath + ".pkl"}')
        self.data.to_json(self.outpath + '.jsonl', orient='records', lines=True)
        print(f'Saved output to {self.outpath + ".jsonl"}')

