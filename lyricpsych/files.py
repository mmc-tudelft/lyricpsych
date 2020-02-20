import os
import pkg_resources


LIWC = os.environ['LIWC_PATH']  # this is required to be set
HEXACO = 'data/hexaco.json'
PERS_ADJ = 'data/personality_adjectives.json'
VAL_WORDS = 'data/value_inventory_Wilson18.json'
MXM2MSD = 'data/mxm2msd.txt'


__all__ = [
    'hexaco',
    'personality_adj',
    'value_words',
    'liwc_dict',
    'mxm2msd'
]


def hexaco():
    """ Get the path to hexaco inventry texts
    
    The reference of the text is as follows:
    
        http://hexaco.org/scaledescriptions
        
    Returns:
        str: filename of hexaco inventory
    """
    return pkg_resources.resource_filename(__name__, HEXACO)


def personality_adj():
    """ Read the filename of personality
    
    Returns:
        str: filename of personality adjectives
    """
    return pkg_resources.resource_filename(__name__, PERS_ADJ)
    
    
def value_words():
    """ Read the filename of values 
    
    Returns:
        str: filename of value words 
    """
    return pkg_resources.resource_filename(__name__, VAL_WORDS)

    
def liwc_dict():
    """ Read the filename of LIWC dictionary
    
    Returns:
        str: filename of LIWC dictionary
    """
    return pkg_resources.resource_filename(__name__, LIWC)

    
def mxm2msd():
    """ Read the filename of map between MxM and MSD
    
    Returns:
        str: filename of map between MxM and MSD
    """
    return pkg_resources.resource_filename(__name__, MXM2MSD)