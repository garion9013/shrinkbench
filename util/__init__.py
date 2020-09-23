from .automap import AutoMap
from .color import printc, highlight, colors
from .csvlogger import CSVLogger
from .online import OnlineStats, OnlineStatsMap
import json

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False
