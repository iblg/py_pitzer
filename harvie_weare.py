import numpy as np
from scipy.constants import *

class Harvie_Weare_Salt:
    def __init__(self, T = 25, vcat = 1, van = 1, zcat = 1, zan = -1, mlimit = 6.144, alpha1 = 1, alpha2 = 0, beta0 = 0.07722, beta1 = 0.25183, beta2 = 0, Cphi = 0.00106):
        self.T = T
        if T == 25:
            self.A = 0.392
            self.b = 1.2
        else:
            print('Temperature is not 25 C. '
                  'Debye-Hueckel constant A is calculated assuming T == 25 in code. This will introduce errors.')
            print('b parameter is 1.2 at 25 C. I don\'t know what it is at other temperatures.')

        self.vcat = vcat
        self.van = van
        self.v = vcat + van

        self.zcat = zcat
        self.zan = zan

        self.m = np.linspace(10**-5, mlimit, 1000)
        self.Z = self.get_Z()
        self.I = self.get_I()

        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.Cphi = Cphi
        self.C = self.get_C()

        self.Bphi = self.get_Bphi()

        self.Phi = self.get_Phi()

        return


    def get_Z(self):
        Z = self.m * self.vcat * np.abs(self.zcat) + self.m * self.van * np.abs(self.zan)
        return Z

    def get_I(self):
        I = self.m * self.vcat * self.zcat**2 + self.m * self.van * self.zan**2
        return 0.5 * I

    def get_C(self):
        C = self.Cphi/(2 * np.sqrt(np.abs(self.zcat * self.zan)))
        return C

    def get_Bphi(self):
        Bphi = self.beta0 + self.beta1 * np.exp(- self.alpha1 * self.I**0.5) + self.beta2 * np.exp(- self.alpha2 * self.I**0.5)
        return Bphi

    def get_Phi(self):
        phi = self.Bphi + self.Z * self.C
        phi = phi * self.m**2
        newterm = - self.A * self.I**1.5 / (1 + self.b * self.I**0.5)
        phi += newterm
        phi = phi * 2 / (self.m * self.v)
        phi += 1
        return phi



