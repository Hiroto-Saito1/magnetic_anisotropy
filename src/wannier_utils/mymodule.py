#!/usr/bin/env python

"""
Description:
    Original general functions and classes.

Usage:
    mymodule.py [-h | --help]

Options:
    -h --help   Show this help screen.

Author:
    Katsuhiro Arimoto
"""

from logging import FileHandler, Formatter, getLogger, Logger, StreamHandler, \
     DEBUG, INFO, WARNING, ERROR, CRITICAL
from pathlib import Path
from subprocess import run, PIPE
from sys import stdout
from typing import Any, Optional, Union

import numpy as np


#--- function returning original logger ---
# This function must be located at the top to define the variable "logger".
# reference: https://zenn.dev/wtkn25/articles/python-logging
def get_my_logger(
    name: str, 
    level: str = "debug", 
    is_log_file: bool = False, 
    log: Optional[Path] = None, 
) -> Logger:
    levels = {
        "debug": DEBUG, 
        "info": INFO, 
        "warning": WARNING, 
        "error": ERROR, 
        "critical": CRITICAL, 
    }
    logger = getLogger(name); logger.setLevel(levels[level.lower()])
    if is_log_file and log:
        handler = FileHandler(log)
    else:
        handler = StreamHandler(stream=stdout)
    handler.setLevel(levels[level.lower()])
    formatter = Formatter("%(asctime)s [%(name)s %(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_my_logger(__name__)


#--- function converting a numpy object to a Python object ---
def convert_numpy_object(obj: Any) -> Union[int, float, complex, bool, np.ndarray]:
    if isinstance(obj, np.ndarray):
        return obj.astype("O")
    if isinstance(obj, (int, float, complex, np.bool_, np.str_)):
        arr: np.ndarray = np.array([obj])
        return arr.astype("O")[0]
    raise ValueError("Given object does not match to NumPy data types.")
