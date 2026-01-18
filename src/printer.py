from __future__ import annotations
from enum import Enum
import os
import sys


def get_env_flag(name: str, default: bool = False) -> bool:
    match os.getenv(name):
        case "1":
            return True
        case "0":
            return False
        case _:
            return default


class Config:
    banner: bool
    log: bool
    colors: bool


CONFIG = Config()
CONFIG.banner = get_env_flag("PRIMAL_BANNER", default=True)
CONFIG.log = get_env_flag("PRIMAL_LOG", default=True)
CONFIG.colors = get_env_flag("PRIMAL_COLORS", default=True)


class TerminalControl:
    RESET = "\033[0m"
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class Style(Enum):
    LOG = 0
    BANNER = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    SUCCESS = 5
    FAIL = 6


class Printer:
    def __init__(self, style: Style):
        match style:
            case Style.LOG:
                self.color = TerminalControl.GRAY
                self.output = sys.stdout
                self.enabled = CONFIG.log
            case Style.BANNER:
                self.color = TerminalControl.WHITE
                self.output = sys.stdout
                self.enabled = CONFIG.banner
            case Style.INFO:
                self.color = TerminalControl.CYAN
                self.output = sys.stdout
                self.enabled = True
            case Style.WARNING:
                self.color = TerminalControl.YELLOW
                self.output = sys.stderr
                self.enabled = True
            case Style.ERROR:
                self.color = TerminalControl.RED
                self.output = sys.stderr
                self.enabled = True
            case Style.SUCCESS:
                self.color = TerminalControl.GREEN
                self.output = sys.stdout
                self.enabled = True
            case Style.FAIL:
                self.color = TerminalControl.MAGENTA
                self.output = sys.stdout
                self.enabled = True

    def __enter__(self) -> Printer:
        if CONFIG.colors:
            self(self.color, end="")
        return self

    def __call__(self, *args, **kwargs):
        if self.enabled:
            print(*args, **kwargs, file=self.output)

    def __exit__(self, exc_type, exc_value, traceback):
        if CONFIG.colors:
            self(TerminalControl.RESET, end="")
        self(end="", flush=True)
