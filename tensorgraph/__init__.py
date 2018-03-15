# import json
# from os.path import dirname
#
# with open(dirname(__file__) + '/pkg_info.json') as fp:
#     _info = json.load(fp)
# __version__ = _info['version']

__version__ = "4.4.12"

from .stopper import EarlyStopper
from .sequential import Sequential
from .graph import Graph
from .node import StartNode, HiddenNode, EndNode
from .progbar import ProgressBar
from .data_iterator import SequentialIterator, StepIterator, SimpleBlocks, DataBlocks
from . import cost
from . import utils
from .dataset.preprocess import *
