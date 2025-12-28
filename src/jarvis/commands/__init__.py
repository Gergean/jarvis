"""Commands for the Jarvis trading system."""

from jarvis.commands.download import download
from jarvis.commands.paper import paper_info, paper_init, paper_list, paper_trade
from jarvis.commands.pinescript import pinescript
from jarvis.commands.plot import plot
from jarvis.commands.status import status
from jarvis.commands.test import test
from jarvis.commands.trade import trade
from jarvis.commands.train import train

__all__ = [
    "download",
    "paper_info",
    "paper_init",
    "paper_list",
    "paper_trade",
    "pinescript",
    "plot",
    "status",
    "test",
    "trade",
    "train",
]
