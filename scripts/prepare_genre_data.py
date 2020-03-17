from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import h5py
import numpy as np
from lyricpsych.data import load_mxm2msd


def main(msd_tagtraum_fn, mxm_h5_fn, out_fn):
    """
    """
    # load relevant data
    mxm2msd = load_mxm2msd()
    msd2mxm = {v:k for k, v in mxm2msd.items()}
    
    with h5py.File(mxm_h5_fn, 'r') as hf:
        mxm_tids = [mxm2msd[tid] for tid in hf['features']['ids'][:]]
    mxm_tids_set = set(mxm_tids)

    with open(msd_tagtraum_fn, 'r') as f:
        genre = [l.strip().split('\t') for l in f if l[0] != '#']
        
    # filter out songs that are included in the MxM
    genre = [(tid, g) for tid, g in genre if tid in mxm_tids_set]
    
    # save to the file
    genres = np.array(
        [g for tid, g in genre], dtype=h5py.special_dtype(vlen=str)
    )
    tids = np.array(
        [tid for tid, g in genre], dtype=h5py.special_dtype(vlen=str)
    )
    with h5py.File(out_fn, 'w') as hf:
        hf.create_dataset('tracks', data=tids)
        hf.create_dataset('genre', data=genres)
        
        
if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('msd_tagtraum_fn', type=str,
                        help='path where all genre (MSD-TagTraum) data is stored')
    parser.add_argument('mxm_h5_fn', type=str,
                        help='filename to the song-level lyric features')
    parser.add_argument('out_fn', type=str,
                        help='path to dump processed h5 file')
    args = parser.parse_args()
    
    # run
    main(args.msd_tagtraum_fn, args.mxm_h5_fn, args.out_fn)