from os.path import join, basename, dirname, splitext, exists
import argparse
import logging

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

    # most of the features below should be going to be separated
    # as the subcommand, to handle each of hyper paramters more elegantly
    # for now it's just quick-and-dirty implementation for internal use
    parser.add_argument('--linguistic', dest='linguistic', action='store_true')
    parser.add_argument('--liwc', dest='liwc', action='store_true')
    parser.add_argument('--value', dest='value', action='store_true')
    parser.add_argument('--personality', dest='personality', action='store_true')
    parser.add_argument('--topic', dest='topic', action='store_true')
    parser.set_defaults(
        linguistic=False, liwc=False, value=False,
        personality=False, topic=False
    )
    parser.add_argument('--inventory', dest='inventory', type=str, default=None,
                        help=('filename contains dictionary contains category-words pair'
                              ' that is used for the target inventory'))
    return parser.parse_args()


def extract():
    """command line toolchain for feature extraction
    """
    args = extract_argparse()
    flags = {
        'linguistic': args.linguistic,
        'liwc': args.linguistic,
        'value': args.value,
        'personality': args.value,
        'topic': args.topic,
        'deep': False
    }
    ext_all = False if any([v for v in flags.values()]) else True
    if ext_all:
        flags = {k:True for k in flags.keys()}

    # setup custom inventory
    if args.inventory is None:
        custom_inven = None
    elif not exists(args.inventory):
        raise ValueError('[ERROR] inventory file not found!')
    else:
        custom_inven = json.load(open(args.inventory))

    # 1. loading the data 
    texts = [line.strip() for line in open(args.text)]
    ids = list(range(len(texts)))
    corpus = Corpus(ids, texts, filter_thresh=None,
                    filt_non_eng=True)

    # 2. initiate the extractor
    extractor = TextFeatureExtractor(args.w2v)
    if custom_inven is not None:
        extractor.psych_inventories['inventory'] = custom_inven
        flags['inventory'] = True

    # 3. run the extraction
    features = _extract(corpus, extractor, config=flags)

    # 5. save the file to the disk
    out_fn = join(
        args.out_path,
        splitext(basename(args.text))[0] + '_feat' + EXTENSIONS[args.format]
    )
    SAVE_FN[args.format](features, out_fn)
