import pkg_resources


HEXACO = 'data/hexaco.json'
PERS_ADJ = 'data/personality_adjectives.csv'
MXM2MSD = 'data/mxm2msd.txt'


__all__ = [
    'hexaco',
    'personality_adj',
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
    
    
def mxm2msd():
    """ Read the filename of map between MxM and MSD
    
    Returns:
        str: filename of map between MxM and MSD
    """
    return pkg_resources.resource_filename(__name__, MXM2MSD)