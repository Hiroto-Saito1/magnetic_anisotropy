#!/usr/bin/env python

"""
Description:
    Prepare Logger instance.
"""

from logging import FileHandler, Formatter, getLogger, Logger, StreamHandler, \
     DEBUG, INFO, WARNING, ERROR, CRITICAL
from pathlib import Path
from typing import Optional
from sys import stdout


def get_logger(
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
