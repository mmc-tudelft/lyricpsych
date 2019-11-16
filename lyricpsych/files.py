import pkg_resources


HEXACO = 'data/hexaco.json'


__all__ = [
    'hexaco'
]


def hexaco():
    """ Get the path to hexaco inventry texts
    
    The reference of the text is as follows:
    
        http://hexaco.org/scaledescriptions
        
    Returns:
        str: filename of hexaco inventory
    """
    return pkg_resources.resource_filename(__name__, HEXACO)