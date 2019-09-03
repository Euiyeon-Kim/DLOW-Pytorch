'''
    Have to open visdom server first
    'python -m visdom.server' or 'visdom'
'''

import sys
from visdom import Visdom
from . import utils

sys.path.append("..")
