from enum import Enum


class Color(Enum):
    RED = "RED"
    GREEN = "GREEN"


class ActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    STAY = "STAY"
    ERR = "ERR"
