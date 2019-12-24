from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import pandas as pd
import numpy as np
import h5py
from scipy import sparse as sp
import sqlite3
from lyricpsych.files import mxm2msd as mxm2msd_fn


def main(msd_lastfm_fn, mxm_h5_fn, out_fn, n_top_tags=50):
    """
    """
    # load relevant data
    with open(mxm2msd_fn()) as f:
        mxm2tid = dict([line.strip('\n').split(',') for line in f])
    tid2mxm = {tid:mxm for mxm, tid in mxm2tid.items()}

    with h5py.File(mxm_h5_fn, 'r') as hf:
        mxm_tids = [mxm2tid[tid] for tid in hf['features']['ids'][:]]

    with sqlite3.connect(msd_lastfm_fn) as conn:
        cur = conn.cursor()
        tid_tag = [
            (r[0]-1, r[1]-1, r[2]) for r in
            cur.execute('SELECT * FROM tid_tag').fetchall()
        ]
        tids = [r[0] for r in cur.execute('SELECT * FROM tids').fetchall()]
        tags = [r[0] for r in cur.execute('SELECT * FROM tags').fetchall()]

    # filter out tracks that matches to MxM
    target_tids = set(mxm_tids).intersection(set(tids))
    target_indices = {i for i, tid in enumerate(tids) if tid in target_tids}
    tid_tag = [r for r in tid_tag if r[0] in target_indices]
     
    # filter out tracks that don't belong to the top-50 list
    tag_pop = {}
    for row in tid_tag:
        if row[1] not in tag_pop:
            tag_pop[row[1]] = 1
        else:
            tag_pop[row[1]] += 1

    top_tags = [
        t for t, c in
        sorted(tag_pop.items(), key=lambda x:-x[1])[:n_top_tags]
    ]
    top_tags_set = set(top_tags)
    tid_tag = [r for r in tid_tag if r[1] in top_tags_set]
    
    # build track-tag matrix
    unique_tracks = {t:i for i, t in enumerate(target_indices)}
    unique_tags = {t:i for i, t in enumerate(top_tags)}
    tid_tag = [(unique_tracks[r[0]], unique_tags[r[1]], r[2]) for r in tid_tag]
    
    I, J, V = zip(*tid_tag)
    Y = sp.coo_matrix((V, (I, J))).tocsr()
    unique_tags = {i:tags[t] for t, i in unique_tags.items()}
    unique_tracks = {i:tids[t] for t, i in unique_tracks.items()}
    
    # save to the file
    track_list = np.array(
        [unique_tracks[i] for i in range(Y.shape[0])],
        dtype=h5py.special_dtype(vlen=str)
    )
    tag_list = np.array(
        [unique_tags[i] for i in range(Y.shape[1])],
        dtype=h5py.special_dtype(vlen=str)
    )
    with h5py.File(out_fn, 'w') as hf:
        hf.create_dataset('data', data=Y.data)
        hf.create_dataset('indices', data=Y.indices)
        hf.create_dataset('indptr', data=Y.indptr)
        hf.create_dataset('tracks', data=track_list)
        hf.create_dataset('tags', data=tag_list)
        
        
if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('msd_lastfm_fn', type=str,
                        help='path where all autotagging (MSD-LastFM) data is stored')
    parser.add_argument('mxm_h5_fn', type=str,
                        help='filename to the song-level lyric features')
    parser.add_argument('out_fn', type=str,
                        help='path to dump processed h5 file')
    parser.add_argument('--n-top-tags', default=50, type=int,
                        help='popularity based cutoff')
    args = parser.parse_args()
    
    # run
    main(args.msd_lastfm_fn, args.mxm_h5_fn, args.out_fn, args.n_top_tags)
        
        