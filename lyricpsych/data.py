from os.path import basename, join
import glob
import json
from tqdm import tqdm

from .files import hexaco

INVENTORIES = {
    'hexaco': hexaco()
}


def load_mxm_lyrics(fn):
    """ Load a MusixMatch api response
    
    Read API (track_lyrics_get_get) response.
    
    Inputs:
        fn (str): filename
        
    Returns:
        list of string: lines of lyrics
        string: musixmatch tid
    """
    d = json.load(open(fn))['message']
    header, body = d['header'], d['body']
    
    status_code = header['status_code']
    lyrics_text = []
    tid = basename(fn).split('.json')[0]
    
    if status_code == 200.:
        if body['lyrics']:
            lyrics = body['lyrics']['lyrics_body'].lower()
            if lyrics != '':
                lyrics_text = [
                    l for l in lyrics.split('\n') if l != ''
                ][:-3]
                
    return tid, ' '.join(lyrics_text)


def load_lyrics_db(path, fmt='json', verbose=True):
    """ Load loyrics db (crawled) into memory
    
    Inputs:
        path (string): path where all the api responses are stored
        fmt (string): format of which lyrics are stored
        verbose (bool): indicates whether progress is displayed
    
    Returns:
        list of tuple: lyrics data
    """
    return [
        load_mxm_lyrics(fn)
        for fn in tqdm(
            glob.glob(join(path, '*.{}'.format(fmt))),
            disable=not verbose, ncols=80
        )
    ]


def load_inventory(inventory='hexaco'):
    """ Load psych inventory to be used as target
    
    Inputs:
        inventory (string): type of inventory to be loaded {'hexaco', 'value'}
    
    Outputs:
        list of tuple: inventory data
    """
    if inventory not in INVENTORIES:
        raise ValueError('[ERROR] {} is not supported!'.format(inventory))
        
    y = json.load(open(INVENTORIES[inventory]))['inventory']
    return list(y.items()) 