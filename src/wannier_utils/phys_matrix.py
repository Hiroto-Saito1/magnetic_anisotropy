#!/usr/bin/env python

from itertools import product

import numpy as np

from wannier_utils.wannier_system import WannierSystem


class PhysMatrix:
    """
    calculate one particle quantities.

    Attributes:
        ws: WannierSystem instance.
    """
    def __init__(self, ws: WannierSystem):
        self.ws = ws

    def spin_matrices(self):
        """
        define spin matrix in Wannier gauge for one-shot or symmetry adapted case.

        Returns:
            spin: spin matrices in numpy array of (4, num_wann, num_wann) shape.
        """
        assert not self.ws.num_wann%2, "the number of Wannier orbitals is odd"
        spin = np.zeros((4, self.ws.num_wann, self.ws.num_wann), dtype=np.complex128)
        paulis = [
            np.identity(2, dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            np.array([[1, 0], [0, -1]], dtype=np.complex128),
        ]
        for i in range(4):
            spin[i] = np.kron(np.identity(self.ws.num_wann//2), paulis[i])
        return spin

    def orbital_matrices(self):
        """
        define orbital matrix in Wannier gauge.

        Returns:
            lorb: orbital matrices in numpy array of (4, num_wann, num_wann) shape.

        Todo:
            Implement p orbitals.
        """
        lorb = np.zeros([4, self.ws.num_wann, self.ws.num_wann], dtype=np.complex128)
        lp = np.zeros([4, 3, 3], dtype=np.complex128)
        lp[0] = np.array([                  # Lx
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
        ], dtype=np.complex)
        lp[1] = np.array([                  # Ly
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.complex)/np.sqrt(2)
        lp[2] = np.array([                  # Lz
            [0, -1j, 0],
            [1j, 0, -1j],
            [0, 1j, 0],
        ])/np.sqrt(2)

        ld = np.zeros([4, 5, 5], dtype=np.complex128)
        ld[0] = np.identity(5, dtype=np.complex128)
        ld[1] = np.array([                  # Lx
            [0, 0, 1j*np.sqrt(3), 0, 0], 
            [0, 0, 0, 0, 1j], 
            [-1j*np.sqrt(3), 0, 0, -1j, 0], 
            [0, 0, 1j, 0, 0], 
            [0, -1j, 0, 0, 0],
        ])
        ld[2] = np.array([                  # Ly
            [0, -1j*np.sqrt(3), 0, 0, 0], 
            [1j*np.sqrt(3), 0, 0, -1j, 0], 
            [0, 0, 0, 0, -1j], 
            [0, 1j, 0, 0, 0], 
            [0, 0, 1j, 0, 0],
        ])
        ld[3] = np.array([                  # Lz
            [0, 0, 0, 0, 0],
            [0, 0, -1j, 0, 0],
            [0, 1j, 0, 0, 0],
            [0, 0, 0, 0, -2j],
            [0, 0, 0, 2j, 0],
        ])

        for p, q in product(range(self.ws.num_wann), repeat=2):
            if self.ws.nnkp.nw2n[p] != self.ws.nnkp.nw2n[q]: continue   # different atom
            if self.ws.nnkp.nw2l[p] != self.ws.nnkp.nw2l[q]: continue   # different l
            if p % 2 != q % 2: continue                                 # different spin
            if (self.ws.nnkp.nw2l[p] == 0): # s orbital
                lorb[0, p, q] = 0
            if (self.ws.nnkp.nw2l[p] == 1): # p orbital, pz px py
                lorb[1, p, q] = 0
            if (self.ws.nnkp.nw2l[p] == 2): # d orbital: dz2 dxz dyz dx2my2 dxy
                lorb[:, p, q] = ld[:, self.ws.nnkp.nw2m[p] - 1, self.ws.nnkp.nw2m[q] - 1]
        return lorb

    def matrix_for_each_atom(self, mat: np.ndarray):
        """
        devide matrix into each atom component.

        Args:
            mat: (num_wann, num_wann) matrix.

        Returns:
            mat_list: matrices projected to each atom.
        """
        mat_list = []
        for n in self.ws.natom:
            d = np.diag([1 if self.ws.nnkp.nw2n[p] == n else 0 for p in range(self.ws.num_wann)])
            mat_list.append(np.einsum("ij,jk,kl->il", d, mat, d))
        return mat_list
