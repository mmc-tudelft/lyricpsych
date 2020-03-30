from os.path import join, basename, dirname, splitext, exists
import argparse

from .pipelines import TextFeatureExtractor
from .data import Corpus


K = 25
THRESH = [5, 0.3]


def get_argparser():
    """
    """
    parser = argparse.ArgumentParser(
        description="Extracts textual feature using various psych inventories"
    )
    parser.add_argument('text', type=str,
                        help='filename for the file contains text. one sentence per line')
    parser.add_argument('out_path', type=str,
                        help='path where the output (.csv) is stored')

    parser.add_argument('--w2v-fn', default=None, type=str,
                        help='path where the gensim w2v model is stored')

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


def main():
    """command line toolchain main
    """
    # process the arguments
    args = setup_argparser()
    flags = {
        'linguistic': args.linguistic,
        'liwc': args.linguistic,
        'value': args.value,
        'personality': args.value,
        'topic': args.topic
    }
    ext_all = False if any([v for v in flags.values()]) else True
    if ext_all:
        flags = {k:True for k in flags.keys()}

    if not exists(args.inventory):
        raise ValueError('[ERROR] inventory file not found!')
    elif args.inventory is None:
        custom_inven = None
    else:
        custom_inven = json.load(open(args.inventory))

    try:
        # loading the data 
        texts = [lines.strip() for line in open(args.text)]
        ids = list(range(len(texts)))
        corpus = Corpus(ids, texts, filter_thresh=None,
                        filt_non_eng=True)

        # 2. initiate the extractor
        features = {}
        extractor = TextFeatureExtractor(args.w2v_fn)
        if flags['liwc']:
            features[k] = extractor.liwc(corpus)

        if flags['linguistic']:
            features[k] = extractor.linguistic_features(corpus)

        corpus.filter_thresh = THRESH
        corpus._preproc()

        if flags['personality']:
            pers_feat, pers_cols = extractor._inventory_scores(
                corpus, extractor.psych_inventories['personality']
            )
            features['personality'] = TextFeature(
                'personality', corpus.ids, pers_feat, pers_cols
            )

        if flags['value']:
            val_feat, val_cols = extractor._inventory_scores(
                corpus, extractor.psych_inventories['value']
            )
            features['value'] = TextFeature(
                'value', corpus.ids, val_feat, val_cols
            )

        if custom_inven is not None:
            features['inventory'] = extractor._inventory_scores(
                corpus, custom_inven
            )

        if flags['topic']:
            features['topic'] = extractor.topic_distributions(corpus, k=K)

        out_fn = join(
            args.out_path,
            splitext(basename(args.text))[0] + '_feat.csv'
        )

        ids = list(features.values())[0].ids  # anchor
        # first aggregate the data
        agg_feature = []
        agg_feat_cols = []
        for key, feat in features.items():
            x = feat.features[[feat.inv_ids[i] for i in ids]]
            agg_feature.append(x)
            agg_feat_cols.extend(feat.columns)
        agg_feature = np.hstack(agg_feature)

        with open(out_fn, 'w') as f:
            f.write(','.join(agg_feat_cols) + '\n')
            for row in agg_feature:
                f.write(','.join(['{:.8f}'.format(y) for y in row]) + '\n')

    except Exception as e:
        print(e)
