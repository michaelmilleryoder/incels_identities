""" Train word2vec model on incels data 
    @author Michael Miller Yoder
    @date 2023
"""

import os
import pdb
import logging

import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader

from data import DataLoader

gensim.models.word2vec.logger.level = logging.INFO


class Word2vecTrainer:
    """ Trains word2vec embeddings """

    def __init__(self, name, outpath, pretrained_name=None, pretrained_path=None):
        """ Args:
                name: name of the dataset
                outpath: a directly path where trained model and vectors will be saved
                pretained_name: name of the pretrained vectors, to be loaded from the Gensim downloader
                pretrained_path: path to pretrained word vectors to start with. If None, train from scratch.
        """
        self.name = name
        self.outpath = outpath
        self.pretrained_name = pretrained_name
        self.pretrained_path = pretrained_path

    def train(self, data):
        """ Train embeddings, save out.
            Args:
                data: a list of sentences or posts to train on
        """

        # Load pretrained background embeddings
        print("Loading pretrained embeddings...")
        #pretrained_wv = KeyedVectors.load_word2vec_format(self.pretrained_path, binary=True)

        print('Building vocab...')
        model = Word2Vec(vector_size=300, min_count=20, workers=20)
        model.build_vocab(data)
        #model.build_vocab([list(pretrained_wv.key_to_index.keys())], update=True) # should add words, though doesn't seem to
        #model.intersect_word2vec_format(self.pretrained_path, lockf=1.0, binary=True)

        # Train model
        print('Training model...')
        model.train(data, total_examples=len(data), epochs=5)

        # Save model
        print('Saving model...')
        if self.pretrained_name is not None:
            model_name = f'{self.name}_{self.pretrained_name}'
        else:
            model_name = f'{self.name}'
        model_outpath = os.path.join(self.outpath, f'{model_name}.model')
        model.save(model_outpath)
        print(f'Saved model to {model_outpath}')

        # ## Save embeddings in txt format
        print('Saving embeddings...')
        emb_outpath = os.path.join(self.outpath, f'{model_name}.txt')
        model.wv.save_word2vec_format(emb_outpath, binary=False)
        print(f'Saved embeddings to {emb_outpath}')


def main():

    # Settings
    dataset_name = 'incels'
    inpath = '../../data/incels/processed_comments.pkl'
    outpath = '../models/emb/'
    #pretrained_name = 'word2vec-google-news-300'
    #pretrained_path = '../resources/GoogleNews-vectors-negative300.bin'
    pretrained_name = None
    pretrained_path = None

    # Load incels data (goes into RAM, could use Gensim LineSentence to load iteratively from file)
    data = DataLoader(dataset_name, inpath).load()['content'].str.split()
    trainer = Word2vecTrainer(dataset_name, outpath, pretrained_name, pretrained_path)
    trainer.train(data)


if __name__ == '__main__':
    main()
