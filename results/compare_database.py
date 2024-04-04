#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""zodiacで、旧データベース mag20230425 中の result/mpall*.toml 内の finished_wannier と、
データベース mag20231002 中の results/ns*.toml 内の unfinished_wannier の共通部分を返す。

Example:
    >>> python compare_database.py
"""

import numpy as np
import toml

finished_wannier = []
mp_folders = []
for i in range(9):
    with open(
        "../../../mag20231002/MAE_programs/results/mpall{}.toml".format(i + 1), "r"
    ) as f:
        result = toml.load(f)
        finished_wannier.extend(result["finished_wannier"])
        mp_folders.extend(result["mp_folders"])
print("Num of finished_wannier in old database: {}".format(len(finished_wannier)))


unfinished_wannier = []
for i in range(11):
    with open("ns{}.toml".format(i + 1), "r") as f:
        result = toml.load(f)
        unfinished_wannier.extend(result["unfinished_wannier"])
print("Num of unfinished_wannier in new database: {}".format(len(unfinished_wannier)))


def intersection(lst1, lst2):
    set1 = set(lst1)
    set2 = set(lst2)
    return list(set1.intersection(set2))


common_group = intersection(finished_wannier, unfinished_wannier)
pick_upped = intersection(mp_folders, unfinished_wannier)
print("Num of common mps: {}".format(len(common_group)))
print("Num of pick-upped mps: {}".format(len(pick_upped)))
with open("compare_database.toml", "w") as f:
    result = {"unfinished mps": common_group, "pick-upped mps": pick_upped}
    toml.dump(result, f)
