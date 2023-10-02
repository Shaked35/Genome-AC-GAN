import argparse

import yaml

from genome_ac_gan_training import train_genome_ac_model


def load_yaml_to_dict(path):
    with open(path, 'r') as f:
        yaml_dict = yaml.safe_load(f)
        return yaml_dict


def main(path):
    train_genome_ac_model(**load_yaml_to_dict(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with Yaml configuration')
    parser.add_argument('--path', type=str, help='path to yaml file configuration')
    args = parser.parse_args()
    main(path=args.path)
