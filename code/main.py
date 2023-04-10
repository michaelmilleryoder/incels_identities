import argparse
import yaml

from extract_identities import IdentityExtractor
from identity_actions_attributes import ActionAttributeExtractor

from data import Dataset


class DataProcessor:
    """ Run different processing on dataset """

    def __init__(self, config):
        """ Args:
                config: config dictionary
        """
        self.config = config
        self.dataset = None

    def process(self):
        # Load data
        self.dataset = Dataset(self.config['dataset']['name'], self.config['dataset']['inpath'], 
            outpath=self.config['dataset']['outpath'], tokenize=self.config['dataset'].get('tokenize', False), 
            text_column=self.config['dataset']['text_column'],
            vocab_path=self.config['dataset']['vocab_path'])
        self.dataset.load()

        # Extract identities
        if self.config['run']['identity_extraction']:
            identity_extractor = IdentityExtractor(self.dataset, self.config['identities']['name'], 
                self.config['identities']['path'], 
                identities_exclude_path = self.config['identities']['exclude_path'], 
                identities_include_path = self.config['identities']['include_path'],
                load_vocab = self.config['identities']['load_vocab'])
            identity_extractor.extract()

        # Get actions and attributes
        if self.config['run']['action_attribute_extraction']:
            action_attribute_extractor = ActionAttributeExtractor(self.dataset)
            action_attribute_extractor.extract()


def main():

    # Load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', nargs='?', type=str, help='file path to YAML config file')
    args = parser.parse_args()
    with open(args.config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    processor = DataProcessor(config)
    processor.process()

if __name__ == '__main__':
    main()
