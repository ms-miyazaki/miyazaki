'''
予約情報受領ファイルをperson_tableと同じ形式にする
'''

import re
import shlex
from heapq import heapify, heappush, heappop
from typing import List, Tuple
from enum import Enum
from datetime import date

import dascan
import numpy as np
from pydantic import BaseModel
import pandas as pd
import numpy as np

month_list = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
'''
for month in range(1,13):
    month_list.append(str(month) + '月')
'''
MORNING_START =
MORNING_END =
AFTERNON_START =
AFTERNOON_END =

# mapで全てのセルに変換を施す
convert_symbol_dict = {'◯': 1, '●': 1, '✕:': 0, '✖': 0, '×': 0,
                       ';': ':'}


def covert_time_time(t: str) -> str:
    if t.__contains__('時台'):
        HH = t.replace('時台')
        return f'{HH}:00 - {int(HH) + 1}:00'
    else:
        return t


def covert_time(t: str) -> str:
    if t in ['午前中', '午前']:
        return MORNING_START + '-' + MORNING_END
    elif t in ['午後']:
        return AFTERNON_START + '-' + AFTERNOON_END
    else:
        return t
def