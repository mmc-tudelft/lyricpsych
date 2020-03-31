from os.path import join, basename, dirname, splitext, exists, split
import argparse
import logging
import json

import numpy as np

from .feature import TextFeature
from .pipelines import TextFeatureExtractor
from .pipelines import extract as _extract
from .data import Corpus
from .utils import save_feature_h5, save_feature_csv


K = 25
THRESH = [5, 0.3]
EXTENSIONS = {
    'csv': '.csv',
    'hdf5': '.h5'
}
SAVE_FN = {
    'csv': save_feature_csv,
    'hdf5': save_feature_h5
}


class FeatureSetAction(argparse.Action):
    CHOICES = {'linguistic', 'liwc', 'value', 'personality', 'topic'}
    def __call__(self, parser, namespace, values, option_string=None):
        if values:
            for value in values:
                if value not in self.CHOICES:
                    message = ("invalid choice: {0!r} (choose from {1})"
                               .format(value,
                                       ', '.join([repr(action)
                                                  for action in self.CHOICES])))
                    raise argparse.ArgumentError(self, message)
            setattr(namespace, self.dest, values)


def extract_argparse():
    # process the arguments
    parser = argparse.ArgumentParser(
        description="Extracts textual feature using various psych inventories"
    )
    parser.add_argument('text', type=str,
                        help='filename for the file contains text. one sentence per line')
    parser.add_argument('out_path', type=str,
                        help='path where the output (.csv) is stored')

    parser.add_argument('--w2v', default=None, type=str,
                        help='name of the gensim w2v model')
    parser.add_argument('--format', default='csv', type=str,
                        choices=set(EXTENSIONS),
                        help='file format to be saved')
    parser.add_argument('--inventory', dest='inventory', type=str, default=None,
                        help=('filename contains dictionary contains category-words pair'
                              ' that is used for the target inventory'))
    parser.add_argument('--features', nargs="*", action=FeatureSetAction,
                        default=['linguistic', 'personality', 'value', 'topic'],
                        help='indicators for the desired featuresets')
    return parser.parse_args()


def extract():
    """command line toolchain for feature extraction
    """
    args = extract_argparse()
    flags = set(args.features)

    # 1. loading the data 
    texts = [line.strip() for line in open(args.text)]
    ids = list(range(len(texts)))
    corpus = Corpus(ids, texts, filter_thresh=None,
                    filt_non_eng=True)

    # 2. initiate the extractor
    extractor = TextFeatureExtractor(args.w2v)

    # setup custom inventory
    if args.inventory is not None:
        if not exists(args.inventory):
            raise ValueError('[ERROR] inventory file not found!')
        else:
            custom_inven = json.load(open(args.inventory))
            extractor.psych_inventories['inventory'] = custom_inven
            flags.add('inventory')

    # 3. run the extraction
    features = _extract(corpus, extractor, features=flags)

    # 4. save the file to the disk
    out_fn = join(
        args.out_path,
        splitext(basename(args.text))[0] + '_feat' + EXTENSIONS[args.format]
    )
    SAVE_FN[args.format](features, out_fn)
