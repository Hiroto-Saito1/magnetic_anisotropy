#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""maeの角度依存性をプロットし、フィッティングから磁気異方性定数を求める。

Example:
    >>> python plot.py
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
        self.plot_angle_dep(self.energy)
        self.popt, self.perr = self.fit_angle_dep(self.energy)
        self.plot_phi_dep(self.energy, self.popt)
        self.plot_theta_dep(self.energy, self.popt)
        self.write(self.popt, self.perr)

    def read_angle_dep(self, theta_mesh=45, phi_mesh=45) -> np.ndarray:
        """angle_dep.txt の結果を読み込み、行列の形に返す。"""
        energy = np.zeros([theta_mesh, phi_mesh])
        with open("angle_dep.txt", "r") as f:
            f.readline()
            f.readline()
            for t_i in range(theta_mesh):
                for p_j in range(phi_mesh):
                    energy[t_i, p_j] = float(f.readline())
        return energy

    def plot_angle_dep(self, energy, theta_mesh=45, phi_mesh=45):
        """angle_dep.txt をカラープロットする。"""
        plt.rcParams["font.size"] = 16
        plt.rcParams["figure.subplot.left"] = 0.20
        plt.rcParams["figure.subplot.bottom"] = 0.20
        plt.figure()
        phi = np.linspace(0, 180, phi_mesh)
        theta = np.linspace(0, 180, theta_mesh)
        plt.figure()
        plt.pcolor(phi, theta, (energy - energy[0, 0]) * 10**6, shading="auto")
        plt.xlabel("$\phi$ [deg]")
        plt.ylabel("$\\theta$ [deg]")
        plt.colorbar(label="$dE$ [$\mathrm{\mu eV}/f.u.$]")
        plt.xticks([0, 45, 90, 135, 180], ["0", "45", "90", "135", "180"])
        plt.yticks([0, 45, 90, 135, 180], ["0", "45", "90", "135", "180"])
        plt.grid()
        plt.savefig("angle_dep.pdf")

    def fitting_func(
        self, X, K1: float, K2: float, K3: float, theta_0: float, phi_0: float
    ) -> List:
        """No.123の空間群の場合の、現象論的なエネルギーフィッティング関数。"""
        phi, theta = X
        z = (
            K1 * np.sin((theta + theta_0) * np.pi / 180) ** 2
            + K2 * np.sin((theta + theta_0) * np.pi / 180) ** 4
            + K3
            * np.sin((theta + theta_0) * np.pi / 180) ** 4
            * np.cos(4 * (phi + phi_0) * np.pi / 180)
        )
        return np.ravel(z)

    def fit_angle_dep(self, energy, theta_mesh=45, phi_mesh=45) -> Tuple:
        """フィッティングで得られた磁気異方性定数とその誤差を [ueV] 単位で返す。"""
        popt, pcov = curve_fit(
            self.fitting_func,
            np.meshgrid(np.linspace(0, 180, phi_mesh), np.linspace(0, 180, theta_mesh)),
            np.ravel((energy - energy[0, 0]) * 10**6),
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def plot_phi_dep(
        self, energy, popt, theta_mesh=45, phi_mesh=45, phi_fitted_mesh=10**3
    ):
        """theta = 90 [deg] の phi方向の依存性をプロットする。"""
        energy_fitted = np.zeros(phi_fitted_mesh)
        for i in range(phi_fitted_mesh):
            energy_fitted[i] = self.fitting_func(
                (np.linspace(0, 180, phi_fitted_mesh)[i], 90), *popt
            )
        plt.rcParams["font.size"] = 16
        plt.rcParams["figure.subplot.left"] = 0.20
        plt.rcParams["figure.subplot.bottom"] = 0.20
        plt.figure()
        plt.scatter(
            np.linspace(0, 180, phi_mesh),
            (energy[int(np.floor(theta_mesh / 2)), :] - energy[0, 0]) * 10**6,
            color="black",
        )
        plt.plot(np.linspace(0, 180, phi_fitted_mesh), energy_fitted, color="black")
        plt.grid()
        plt.xticks([0, 45, 90, 135, 180], ["0", "45", "90", "135", "180"])
        plt.xlabel("$\phi$ [deg]")
        plt.ylabel("$dE$ [$\mathrm{\mu eV}/f.u.$]")
        plt.savefig("phi_dep.pdf")

    def plot_theta_dep(
        self, energy, popt, theta_mesh=45, theta_fitted_mesh=10**3, phi_mesh=45
    ):
        """phi = 0 [deg] の theta方向の依存性をプロットする。"""
        energy_fitted = np.zeros(theta_fitted_mesh)
        for i in range(theta_fitted_mesh):
            energy_fitted[i] = self.fitting_func(
                (0, np.linspace(0, 180, theta_fitted_mesh)[i]), *popt
            )
        plt.rcParams["font.size"] = 16
        plt.rcParams["figure.subplot.left"] = 0.20
        plt.rcParams["figure.subplot.bottom"] = 0.20
        plt.figure()
        plt.scatter(
            np.linspace(0, 180, theta_mesh),
            (energy[:, 0] - energy[0, 0]) * 10**6,
            color="black",
        )
        plt.plot(np.linspace(0, 180, theta_fitted_mesh), energy_fitted, color="black")
        plt.grid()
        plt.xticks([0, 45, 90, 135, 180], ["0", "45", "90", "135", "180"])
        plt.xlabel("$\\theta$ [deg]")
        plt.ylabel("$dE$ [$\mathrm{\mu eV}/f.u.$]")
        plt.savefig("theta_dep.pdf")

    def write(self, popt, perr):
        """フィッティングで求められた磁気異方性定数を mae_constants.txt に書き出す。"""
        with open("mae_constants.txt", "w") as f:
            f.write(
                "K1= {} pm {} [ueV]\n".format(popt[0], perr[0])
                + "K2= {} pm {} [ueV]\n".format(popt[1], perr[1])
                + "K3= {} pm {} [ueV]\n\n".format(popt[2], perr[2])
            )
            f.write(
                "K1= {} pm {} [MJ/m^3]\n".format(
                    popt[0] * 7.1202297 * 1e-3, perr[0] * 7.1202297 * 1e-3
                )
                + "K2= {} pm {} [MJ/m^3]\n".format(
                    popt[1] * 7.1202297 * 1e-3, perr[1] * 7.1202297 * 1e-3
                )
                + "K3= {} pm {} [MJ/m^3]\n\n".format(
                    popt[2] * 7.1202297 * 1e-3, perr[2] * 7.1202297 * 1e-3
                )
            )


if __name__ == "__main__":
    Mae_constants()
