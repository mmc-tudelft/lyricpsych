from os.path import join, dirname, basename
import json

from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler
from hyperopt_AT import (MODELS,
                         DATASETS,
                         SEARCH_SPACE,
                         load_data,
                         split_data,
                         prep_data)


def setup_argparser():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str,
                        help='path where all autotagging data is stored')
    parser.add_argument('out_root', type=str,
                        help='path where output is stored')
    parser.add_argument('optres_fn', type=str,
                        help='hyperopt result file (.json)')
    parser.add_argument('model', type=str,
                        choices={'mf', 'mlp'},
                        help='target model to be tested')
    parser.add_argument('dataset', type=str,
                        choices={'msd50', 'msd1k', 'mgt50', 'mgt188'},
                        help='type of dataset')
    parser.add_argument('feature', type=str, choices={'mfcc', 'rand'},
                        help='type of audio feature')
    parser.add_argument('fold', type=int, choices=set(range(10)),
                        help='target fold to be tested')
    parser.add_argument('k', type=int, choices={32, 64, 128},
                        help='model size')
    parser.add_argument('--split-type', type=str, default='nomatch',
                        choices={'klmatch', 'nomatch'},
                        help='split method to be used')
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.set_defaults(scale=False)
    args = parser.parse_args()
    return args


def get_best_model(model, k, dataset, feature, split, seed, optres_fn):
    opt_res = json.load(open(optres_fn))
    best_param = [
        theta for theta in opt_res
        if all([(theta['model'] == model),
                (theta['k'] == k),
                (theta['seed'] == seed),
                (theta['dataset'] == dataset),
                (theta['feature'] == feature),
                (theta['split'] == split)])
    ][0]['params']
    return MODELS[model](k, **best_param)


if __name__ == "__main__":
    args = setup_argparser()
    feats, seeds, labels = prep_data(args)

    model = {
        'seed': get_best_model(
            args.model, args.k, args.dataset, args.feature,
            args.split_type, True, args.optres_fn
        ),
        'noseed': get_best_model(
            args.model, args.k, args.dataset, args.feature,
            args.split_type, False, args.optres_fn
        )
    }
    model['seed'].fit(feats['total_train'], labels['total_train'])
    model['noseed'].fit(feats['total_train'], labels['total_train'])

    scores = {}
    for n_seed in [0, 1, 2, 5]:
        if n_seed == 0:
            m = model['noseed']
        else:
            m = model['seed']

        s = {'auc':{}, 'ndcg':{}}
        for avg in ['samples', 'micro', 'macro']:
            s['auc'][avg] = m.score(
                feats['test'][n_seed], labels['test'][n_seed],
                seeds['test'][n_seed], metric='auc', average=avg
            )

        if args.model != 'mlp':
            for topk in [1, 2, 5, 10, 20]:
                s['ndcg'][topk] = m.score(
                    feats['test'][n_seed], labels['test'][n_seed],
                    seeds['test'][n_seed], metric='ndcg', topk=topk
                )

        scores[n_seed] = s

    # save
    out_fn = join(
        args.out_root,
        '{}{:d}_{}_{}_{}_fold{:d}.json'.format(
            args.model, args.k, args.dataset,
            args.feature, args.split_type, args.fold
        )
    )
    json.dump(scores, open(out_fn, 'w'))
