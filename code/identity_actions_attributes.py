from collections import Counter
import itertools
from multiprocessing import Pool
import pdb

import numpy as np
import spacy
from tqdm import tqdm

from data import Dataset


def unique_term_index(l):
    """ Returns a list of the index of each occurrence of each unique term in the list l """
    ctr = Counter()
    res = []
    for term in l:
        res.append(ctr[term])
        ctr[term] += 1
    return res


def extract_actions_attributes(nlp, text, identity_matches, identity_spans):
    """ Extract actions and attributes based on dependency parse """

    actions_attributes = [] # for each identity mention, {'actions': [actions], {'attributes': [attributes]}

    if len(identity_matches) == 0:
        return actions_attributes

    doc = nlp(text)
    #identity_indexes = unique_term_index(identity_matches)
    tok_offsets = [tok.idx for tok in doc] # character offsets of all tokens

    #for identity, identity_idx in zip(identity_matches, identity_indexes):
    for identity, (beg, end) in zip(identity_matches, identity_spans):
        # Get identity mention locations
        mention_idx = [tok.i for tok in doc if tok.text==identity]
        identity_toks = [tok for tok in doc if tok.idx >= beg and tok.idx + len(tok) <= end]
        #if identity_idx >= len(mention_idx):
        #    continue # probably can't find any matches for the text
        #tok_idx = mention_idx[identity_idx]
        if len(identity_toks) > 1:
            # Choose the head
            unique_heads = [tok for tok in identity_toks if not tok.head in identity_toks]
            if len(unique_heads) != 1:
                pdb.set_trace()
            identity_tok = unique_heads[0]
        elif len(identity_toks) == 0:
            continue # match is within a word, like TODO: fix man matching within 'w*man', which is a mistake
        else:
            identity_tok = identity_toks[0]
        
        # Verbs where identity term was the subject
        verbs = []
        if identity_tok.dep_ == 'nsubj' or identity_tok.dep_ == 'agent':
            verbs.append(identity_tok.head.text)
        #verbs_subj = [tok.head.text for tok in doc if tok.i==tok_idx \
        #    in mention_idx and (tok.dep_=='nsubj' or tok.dep_=='agent')]

        # Verbs where identity term was the object
        if identity_tok.dep_=='dobj' or identity_tok.dep_=='nsubjpass' or \
                identity_tok.dep_=='dative' or identity_tok.dep_=='pobj':
           verbs.append(identity_tok.head.text)
        #verbs_obj = [tok.head.text for tok in doc if tok.i==tok_idx and \
        #    (tok.dep_=='dobj' or tok.dep_=='nsubjpass' or \
        #    tok.dep_=='dative' or tok.dep_=='pobj')]

        # Adjectives that describe the identity term
        adjs = [tok.text.lower() for tok in doc if tok.head == identity_tok and \
            (tok.dep_=='amod' or tok.dep_=='appos' or \
            tok.dep_=='nsubj' or tok.dep_=='nmod')] \
            + [tok.text.lower() for tok in doc if tok.dep_=='attr' and \
                (tok.head.text=='is' or tok.head.text=='was') and \
               any([c==identity_tok for c in tok.head.children])]
        
        actions_attributes.append({'actions': verbs, 'attributes': adjs})

    dep_parse = [tok.dep_ for tok in doc]
    dep_head = [tok.head.text for tok in doc]
    return (actions_attributes, dep_parse, dep_head)


class ActionAttributeExtractor:
    """ Extract verbs and adjectives (actions and attributes) attributed to identity terms """

    def __init__(self, dataset):
        """ Args:
                dataset: Dataset object to do processing on
        """
        self.dataset = dataset
        self.data = self.dataset.data # for ease
        self.nlp = spacy.load('en_core_web_sm', disable=['ner'])

    def extract(self):
        # Process each row to output actions and attributes for each occurrence of identity term
        print("Extracting actions and attributes...")
        zipped = list(zip(itertools.repeat(self.nlp), 
            self.data[self.dataset.text_column], self.data['netmapper_identity_matches'],
            self.data['netmapper_identity_matches_spans']))
        #with Pool(20) as p:
        #    self.data['actions_attributes'], self.data['dep_parse'], self.data['dep_head'] = zip(*p.starmap(
        #        extract_actions_attributes, tqdm(zipped, ncols=80)))
        #    #self.data['actions_attributes'], self.data['dep_parse'], self.data['dep_head'] = zip(*[self.extract_actions_attributes(
        #    #    text, identity_matches) for (text, identity_matches) in tqdm(list(), ncols=80)])
        # for debugging
        self.data['actions_attributes'], self.data['dep_parse'], self.data['dep_head'] = [extract_actions_attributes(*z) for z in tqdm(zipped, ncols=80)]
        self.dataset.save()

