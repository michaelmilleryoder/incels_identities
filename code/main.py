import argparse
import yaml

from extract_identities import IdentityExtractor


def main():

    # Load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', nargs='?', type=str, help='file path to YAML config file')
    args = parser.parse_args()
    with open(args.config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    # Process the data
    extractor = IdentityExtractor(config['dataset']['name'], config['dataset']['inpath'], config['dataset']['outpath'],
        config['identities']['name'], config['identities']['path'], config['dataset']['load_vocab'], 
        config['dataset']['vocab_path'], config['dataset']['text_column'], config['identities']['exclude_path'],
        config['identities']['include_path'])
    extractor.load()
    extractor.extract()


if __name__ == '__main__':
    main()
