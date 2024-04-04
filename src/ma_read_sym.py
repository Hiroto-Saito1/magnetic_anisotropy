# -*- coding: utf-8 -*-

"""pwscf_sym.dat から磁気空間群の対称操作を読み取るプログラム。

Example:
    >>> python src/read_sym.py

Todo:
    * TRS条件をみたす mp-* フォルダの中で、ある空間群操作を持っている (or いない)ものだけ取り出せるような検索機能をつける。
    * inversionのある物質の中だけで、共通の群操作を取り出せるようにする
"""

import os
from pathlib import Path
from typing import Optional, List
import numpy as np
from ma_interface import Ma_interface


class Magnetic_space_group:
    def __init__(
        self,
        num_operations: int,
        name_operations: List[str],
        rotations: np.ndarray,
        translations: np.ndarray,
        time_reversals: int,
        spinor_rotations: np.ndarray,
        num_inv_ope: int,
    ):
        """磁気空間群操作を格納するクラス"""
        self.num_operations = num_operations
        self.name_operations = name_operations
        self.rotations = rotations
        self.translations = translations
        self.time_reversals = time_reversals
        self.spinor_rotations = spinor_rotations
        self.num_inv_ope = num_inv_ope


class Read_sym:
    def __init__(self, root_dir: Optional[Path] = None):
        ma = Ma_interface(root_dir)
        self.ma = ma
        self.sym_dats = self.get_sym_dats_path(ma.mp_folders)
        self.msgs = self.get_msg_operations(self.sym_dats)
        self.get_common_operations(self.msgs)
        # self.get_total_operations(self.msgs)
        # self.get_wo_inversion(self.msgs)
        # self.get_wo_c2z(self.msgs)

    def get_sym_dats_path(self, mp_folders: List[Path]) -> List[Path]:
        """TRSゲージ条件をみたした mp-* フォルダから pwscf_sym.dat のパスを取得し、リストで返す。"""
        sym_dats = []
        for mp in range(len(mp_folders)):
            sym_dats.append(Path(self.ma.find_file_path(mp_folders[mp], "pwscf_sym.dat")))
        return sym_dats

    def get_msg_operations(self, sym_dats: List[Path]) -> List[Magnetic_space_group]:
        """ 磁気空間群操作を各 mp-* フォルダで読み出し、リストで返す。"""
        msgs = []
        for mp in range(len(sym_dats)):
            with open(sym_dats[mp]) as f:
                f.readline()
                f.readline()
                num_operations = int(f.readline())
                name_operations = []
                rotations = np.zeros([num_operations, 3, 3])
                translations = np.zeros([num_operations, 3])
                time_reversals = np.zeros(num_operations, dtype=int)
                spinor_rotations = np.zeros([num_operations, 2, 2], dtype=complex)
                num_inv_ope = np.zeros(num_operations, dtype=int)
                for n_ope in range(num_operations):
                    name = f.readline()[6:-3]
                    formatted_name = name.rstrip()
                    name_operations.append(formatted_name)
                    for col in range(3):
                        _x, _y, _z = map(float, f.readline().split())
                        rotations[n_ope, col, :] = [_x, _y, _z]
                    _x, _y, _z = map(float, f.readline().split())
                    translations[n_ope, :] = [_x, _y, _z]
                    time_reversals[n_ope] = int(f.readline())
                    for col in range(2):
                        for row in range(2):
                            _real, _imag = map(float, f.readline().split())
                            spinor_rotations[n_ope, col, row] = _real + _imag * 1j
                    num_inv_ope[n_ope] = int(f.readline())
            msg = Magnetic_space_group(
                num_operations,
                name_operations,
                rotations,
                translations,
                time_reversals,
                spinor_rotations,
                num_inv_ope,
            )
            msgs.append(msg)
        return msgs

    def get_common_operations(self, msgs: List[Magnetic_space_group]):
        """全フォルダで共通する磁気空間群操作を返す。操作の名前とTRSの有無のタプルの集合を使用する。"""
        common_operations = {
            n_ope
            for n_ope in zip(msgs[0].name_operations, msgs[0].time_reversals)
        }
        for msg in msgs:
            common_operations = common_operations.intersection(
                {n_ope for n_ope in zip(msg.name_operations, msg.time_reversals)}
            )
        print("Num of common operations: {}".format(len(common_operations)))
        print(common_operations)

    def get_total_operations(self, msgs: List[Magnetic_space_group]):
        """含まれる全ての磁気空間群操作を返す。操作の名前とTRSの有無のタプルの集合を使用する。"""
        total_operations = set()
        for msg in msgs:
            total_operations = total_operations.union(
                {n_ope for n_ope in zip(msg.name_operations, msg.time_reversals)}
            )
        print("Num of total operations: {}".format(len(total_operations)))
        print(total_operations)

    def get_wo_inversion(self, msgs: List[Magnetic_space_group]):
        """空間反転操作のない物質のみを取り出す。"""
        mps_wo_inversion = []
        for idx in range(len(msgs)):
            if "inversion" not in msgs[idx].name_operations:
                mps_wo_inversion.append(self.ma.mp_folders[idx])
        print("Num of mps w/o inversion: {}".format(len(mps_wo_inversion)))
        for mp in mps_wo_inversion:
            print(mp.name)

    def get_wo_c2z(self, msgs: List[Magnetic_space_group]):
        """C_2z操作のない物質のみを取り出す。"""
        mps_wo_c2z = []
        for idx in range(len(msgs)):
            if "180 deg rotation - cart. axis [0,0,1]" not in msgs[idx].name_operations:
                mps_wo_c2z.append(self.ma.mp_folders[idx])
        print("Num of mps w/o C_2z: {}".format(len(mps_wo_c2z)))
        self.mps_wo_c2z = mps_wo_c2z
        for mp in mps_wo_c2z:
            print(mp.name)


if __name__ == "__main__":
    # syms = Read_sym(root_dir=Path(os.getcwd() + "/tests"))

    # ns_folders = ["ns1", "ns2", "ns3", "ns4"]
    ns_folders = ["ns11"]
    for ns in ns_folders:
        syms = Read_sym(root_dir=Path(os.getcwd() + "/../" + ns))
