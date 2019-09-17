'''
This module contains some helpful methods tp create insturmetn response functions and 
eventlists according to the open gamma-ray astro data format (gadf)

See  https://gamma-astro-data-formats.readthedocs.io/en/latest/ for some specs. 

'''

from . import time
from . import response
from . import hdus

__all__ = [
    'time',
    'response',
    'hdus'
]
