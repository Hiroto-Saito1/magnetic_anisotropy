#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""maeの角度依存性をプロットし、フィッティングから磁気異方性定数を求める。

Example:
    >>> python mae_constants.py
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from typing import Tuple, List


class Mae_constants:
    def __init__(self):
        """maeの角度依存性を読み、
        プロットし、フィッティングする。
        """
        self.energy = self.read_angle_dep()
        self.theta_fit = self.fit_theta_dep(*self.energy)
        self.phi_fit = self.fit_phi_dep(*self.energy)
        self.plot_phi_dep(*self.energy, *self.phi_fit)
        self.plot_theta_dep(*self.energy, *self.theta_fit)
        self.calc_mae_constants(self.theta_fit, self.phi_fit)

    def read_angle_dep(self, theta_mesh=45, phi_mesh=45) -> Tuple[List]:
        """angle_dep.txt の結果を読み込み、行列の形に返す。"""
        theta_dep = np.zeros(theta_mesh)
        phi_dep = np.zeros(phi_mesh)
        with open("angle_dep_light.txt", "r") as f:
            f.readline()
            f.readline()
            f.readline()
            for t_i in range(theta_mesh):
                theta_dep[t_i] = float(f.readline())
            f.readline()
            for p_j in range(phi_mesh):
                phi_dep[p_j] = float(f.readline())
        return theta_dep, phi_dep

    def fitting_func_theta(
        self,
        X,
        K1: float,
        K23: float,
        theta_0: float,
    ) -> List:
        """No.123の空間群の場合の、現象論的なエネルギーフィッティング関数。"""
        theta = X
        z = (
            K1 * np.sin((theta + theta_0) * np.pi / 180) ** 2
            + K23 * np.sin((theta + theta_0) * np.pi / 180) ** 4
        )
        return z

    def fitting_func_phi(self, X, K12: float, K3: float, phi_0: float) -> List:
        """No.123の空間群の場合の、現象論的なエネルギーフィッティング関数。"""
        phi = X
        z = K12 + K3 * np.cos(4 * (phi + phi_0) * np.pi / 180)
        return z

    def fit_theta_dep(self, theta_dep, phi_dep, theta_mesh=45, phi_mesh=45) -> Tuple:
        """フィッティングで得られた磁気異方性定数とその誤差を [meV] 単位で返す。"""
        popt, pcov = curve_fit(
            self.fitting_func_theta,
            np.linspace(0, 180, theta_mesh),
            (theta_dep - theta_dep[0]) * 10**3,
            p0=[3, 0, 0],
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def fit_phi_dep(self, theta_dep, phi_dep, theta_mesh=45, phi_mesh=45) -> Tuple:
        """フィッティングで得られた磁気異方性定数とその誤差を [meV] 単位で返す。"""
        popt, pcov = curve_fit(
            self.fitting_func_phi,
            np.linspace(0, 180, phi_mesh),
            (phi_dep - theta_dep[0]) * 10**3,
            p0=[3, 0, 0],
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def plot_phi_dep(
        self,
        theta_dep,
        phi_dep,
        popt,
        perr,
        theta_mesh=45,
        phi_mesh=45,
        phi_fitted_mesh=10**3,
    ):
        """theta = 90 [deg] の phi方向の依存性をプロットする。"""
        phi_dep_fitted = np.zeros(phi_fitted_mesh)
        for i in range(phi_fitted_mesh):
            phi_dep_fitted[i] = self.fitting_func_phi(
                np.linspace(0, 180, phi_fitted_mesh)[i], *popt
            )
        plt.rcParams["font.size"] = 16
        plt.rcParams["figure.subplot.left"] = 0.20
        plt.rcParams["figure.subplot.bottom"] = 0.20
        plt.figure()
        plt.scatter(
            np.linspace(0, 180, phi_mesh),
            (phi_dep - theta_dep[0]) * 10**6,
            color="black",
        )
        plt.plot(
            np.linspace(0, 180, phi_fitted_mesh),
            phi_dep_fitted * 10**3,
            color="black",
        )
        plt.grid()
        plt.xticks([0, 45, 90, 135, 180], ["0", "45", "90", "135", "180"])
        plt.xlabel("$\phi$ [deg]")
        plt.ylabel("$dE$ [$\mathrm{\mu eV}/f.u.$]")
        plt.savefig("phi_dep.pdf")

    def plot_theta_dep(
        self,
        theta_dep,
        phi_dep,
        popt,
        perr,
        theta_mesh=45,
        phi_mesh=45,
        theta_fitted_mesh=10**3,
    ):
        """phi = 0 [deg] の theta方向の依存性をプロットする。"""
        theta_dep_fitted = np.zeros(theta_fitted_mesh)
        for i in range(theta_fitted_mesh):
            theta_dep_fitted[i] = self.fitting_func_theta(
                np.linspace(0, 180, theta_fitted_mesh)[i], *popt
            )
        plt.rcParams["font.size"] = 16
        plt.rcParams["figure.subplot.left"] = 0.20
        plt.rcParams["figure.subplot.bottom"] = 0.20
        plt.figure()
        plt.scatter(
            np.linspace(0, 180, theta_mesh),
            (theta_dep - theta_dep[0]) * 10**6,
            color="black",
        )
        plt.plot(
            np.linspace(0, 180, theta_fitted_mesh),
            theta_dep_fitted * 10**3,
            color="black",
        )
        plt.grid()
        plt.xticks([0, 45, 90, 135, 180], ["0", "45", "90", "135", "180"])
        plt.xlabel("$\\theta$ [deg]")
        plt.ylabel("$dE$ [$\mathrm{\mu eV}/f.u.$]")
        plt.savefig("theta_dep.pdf")

    def calc_mae_constants(self, theta_fit, phi_fit):
        """theta方向とphi方向のフィッティング結果から、磁気異方性定数とその誤差を求める"""
        K1 = theta_fit[0][0] * 10**3
        K23 = theta_fit[0][1] * 10**3
        K12 = phi_fit[0][0] * 10**3
        K3 = phi_fit[0][1] * 10**3
        print("MAE constants [ueV]")
        print("K1: {}, K2: {}, K2: {}, K3: {}".format(K1, K23 - K3, K12 - K1, K3))
        K1_err = theta_fit[1][0] * 10**3
        K23_err = theta_fit[1][1] * 10**3
        K12_err = phi_fit[1][0] * 10**3
        K3_err = phi_fit[1][1] * 10**3
        print(
            "K1_err: {}, K2_err: {}, K2_err: {}, K3_err: {}".format(
                K1_err, K23_err - K3_err, K12_err - K1_err, K3_err
            )
        )


if __name__ == "__main__":
    Mae_constants()
