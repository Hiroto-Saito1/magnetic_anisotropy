#!/usr/bin/env python

from typing import List, Optional, Union

import numpy as np

from wannier_utils.mp_points import MPPoints
from wannier_utils.parallel import ParallelK
from wannier_utils.wannier_system import WannierSystem


class WannierKmesh:
    """
    WannierSystem with uniform kmesh.
    """
    def __init__(
        self, 
        ws: WannierSystem, 
        kmesh: List[int], 
        magmoms: Optional[Union[str, np.ndarray]] = None, 
        use_sym: bool = False, 
        nproc: int = 1, 
    ):
        """
        Constructor.

        Args:
            ws (WannierSystem): a WannierSystem instance.
            kmesh (List[int]): numpy array of [nk1, nk2, nk3], or [nk1, nk2, nk3, ks1, ks2, ks3] in fractional coordinates.
            magmoms (Optional[Union[str, np.ndarray]]): magnetic moments to determine magnetic spacegroup symmetry. 
                                                        defaults to None.
            use_sym (bool): whether to reduce k points with spacegroup symmetry. defaults to False.
            nproc (int): the number of process parallelization. defaults to 1.
        """
        self.ws = ws
        self.mp_points = MPPoints(ws, kmesh, magmoms=magmoms, use_sym=use_sym)
        kvec = self.mp_points.irr_kvec if use_sym else self.mp_points.full_kvec
        self.parallel_k = ParallelK(
            ws, 
            kvec, 
            use_sym, 
            self.mp_points.equiv_k, 
            self.mp_points.equiv_sym, 
            nproc, 
        )
