from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import pandas as pd
import h5py
from scipy import sparse as sp
from lyricpsych.files import mxm2msd as mxm2msd_fn


def main(recsys_fn, mxm_h5_fn, msd_tracks_fn, out_fn,
         thresh_ratio=0.5, thresh_num=20):
    """
    """
    # load relevant data
    recsys_data = pd.read_csv(recsys_fn, sep='\t', header=None)
    msd_tracks = pd.read_csv(msd_tracks_fn, sep='<SEP>', header=None)
    mxm2msd = dict([line.strip('\n').split(',') for line in open(mxm2msd_fn())])
    with h5py.File(mxm_h5_fn, 'r') as hf:
        crawled_tid = [mxm2msd[tid] for tid in hf['features']['ids'][:]]

    # index tracks existing both msd and mxm
    n_total_songs = recsys_data.groupby(0)[2].count()
    recsys_data[1] = recsys_data[1].map(dict(msd_tracks[[1, 0]].values))
    recsys_data = recsys_data[recsys_data[1].isin(set(crawled_tid))]
    n_songs_survived = recsys_data.groupby(0)[2].count()
    print(n_total_songs.shape, n_songs_survived.shape)

    # filter out users whose number of survived tracks are too small
    cond_ratio = (n_songs_survived / n_total_songs) > thresh_ratio
    cond_num = n_songs_survived >= thresh_num
    survived_users = n_songs_survived[cond_ratio | cond_num].index
    recsys_data = recsys_data[recsys_data[0].isin(set(survived_users))]
    
    # save to the h5 file format
    users = recsys_data[0].unique()
    items = recsys_data[1].unique()
    users_hash = {u:j for j, u in enumerate(users)}
    items_hash = {i:j for j, i in enumerate(items)}
    recsys_data[0] = recsys_data[0].map(users_hash)
    recsys_data[1] = recsys_data[1].map(items_hash)
    
    V = recsys_data[2].values
    I = recsys_data[0].values
    J = recsys_data[1].values
    X = sp.coo_matrix((V, (I, J)))
    X = X.tocsr()
    
    with h5py.File(out_fn, 'w') as hf:
        hf.create_dataset('indices', data=X.indices)
        hf.create_dataset('indptr', data=X.indptr)
        hf.create_dataset('data', data=X.data)
        hf.create_dataset('users',
                          data=users.astype(
                              h5py.special_dtype(vlen=str)))
        hf.create_dataset('items',
                          data=items.astype(
                              h5py.special_dtype(vlen=str)))
        
        
if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('recsys_fn', type=str,
                        help='path where all recsys (MSD-Echonest) data is stored')
    parser.add_argument('mxm_h5_fn', type=str,
                        help='filename to the song-level lyric features')
    parser.add_argument('msd_tracks_fn', type=str,
                        help='filename to MSD track info (simple version)')
    parser.add_argument('out_fn', type=str,
                        help='path to dump processed h5 file')
    parser.add_argument('--thresh-ratio', default=.9, type=float,
                        help='threshold to filter out robust users (ratio)')
    parser.add_argument('--thresh-num', default=20, type=int,
                        help='threshold to filter out robust users (num)')
    args = parser.parse_args()
    
    # run
    main(
        args.recsys_fn, args.mxm_h5_fn, args.msd_tracks_fn, args.out_fn,
        args.thresh_ratio, args.thresh_num
    )
        
        