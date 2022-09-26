import pandas as pd
import numpy as np

class Pitzer_Salt:
    def __init__(self, mw = 58.44, zcat = 1, zan = 1, vcat = 1, van = 1, mlimit = 6, A = 0.3915, b = 1.2, alpha = 2, B0 = 0.109575, B1 = 1.113, Cphi = 0.0258482):
        """
        Defaults of all parameters are for NaCl
        mw: Molecular weight of salt
        zcat: Charge of cation
        zan: Charge of anion
        vcat: Number of cations when disassociated
        van: Number of anions when disassociated
        mlimit: Limit of solubility, molality (mol/kg water)
        A: Debye-Huckel parameter
        b: some other parameter
        alpha: some other parameter
        B0: first Pitzer parameter
        B1: second Pitzer parameter
        Cphi: third Pitzer parameter
        """
        self.mw = mw
        self.zcat = zcat
        self.zan = zan
        self.vcat = vcat
        self.van = van
        self.v = van + vcat

        self.mlimit = mlimit
        self.m = np.linspace(0.00001, self.mlimit, num = 1000)
        self.xs = self.get_xs()
        self.I = self.get_I()

        self.A = A
        self.b = b
        self.alpha = alpha
        self.B0 = B0
        self.B1 = B1
        self.Cphi = Cphi
        self.Cgamma = 3 * Cphi
        self.CPhi = 2 * Cphi

        self.h = self.get_h()
        self.phi = self.get_phi()
        self.Bphi = self.get_Bphi()
        self.Bgamma = self.get_Bgamma()
        self.Phi = self.get_Phi()
        self.gammapm = self.get_gammapm()

        table = np.concatenate(
            [
                self.m.reshape((-1,1)),
                self.xs.reshape((-1,1)),
                self.I.reshape((-1,1)),
                self.h.reshape((-1,1)),
                self.phi.reshape((-1,1)),
                self.Bgamma.reshape((-1,1)),
                self.Bphi.reshape((-1,1)),
                self.Phi.reshape((-1,1)),
                self.gammapm.reshape((-1,1))
            ],
                               axis = 1)
        self.table = pd.DataFrame(table, columns = ['m','xs', 'I', 'h', 'phi', 'Bgamma', 'Bphi', 'Phi', 'gammapm'])


        return

    def get_xs(self):
        xs = self.v * self.m /(self.v * self.m + 1/(18.015/1000))
        return xs

    def get_I(self):
        I = 0.5 * (self.m * self.vcat * self.zcat**2 + self.m * self.van * self.zan**2)
        return I

    def get_h(self):
        h = np.log(1 + self.alpha * np.sqrt(self.I)) / (2 * self.b)
        return h

    def get_phi(self):
        phi = self.m * 18.015 * 10**(-3)
        return phi

    def get_Bphi(self):
        Bphi = self.B0+self.B1*np.exp(-self.alpha*np.sqrt(self.I))
        # Bphi = self.B0 + self.B1 * np.exp(-self.alpha * np.sqrt(self.I))
        return Bphi

    def get_Bgamma(self):
        # Bgamma = self.B0 + 2 * self.B1 / (self.alpha**2) / self.I * (1 - (1 + self.alpha * np.sqrt(self.I)) * np.exp(-self.alpha * np.sqrt(self.I))) + self.Bphi
        Bgamma = self.B0+2*self.B1/(self.alpha**2)/self.I*(1-(1+self.alpha*np.sqrt(self.I))*np.exp(-self.alpha*np.sqrt(self.I)))+self.Bphi
        return Bgamma

    def get_Phi(self):
        # Phi = (1 - np.abs(self.zcat * self.zan) * self.A * np.sqrt(self.I) / (1 + self.b * np.sqrt(self.I))+self.m * 2 *self.vcat * self.van / self.v * self.Bphi
        #        + 2 * (self.vcat * self.van)**(1.5) / self.v * self.m**2 * self.Cphi)
        Phi = (1
                -np.abs(self.zcat*self.zan)*self.A*np.sqrt(self.I)/(1+self.b*np.sqrt(self.I))
                +self.m* 2 * self.vcat * self.van/self.v*self.Bphi
                + 2*(self.vcat*self.van)**(1.5) / self.v * self.m**2*self.CPhi)
        return Phi
    
    def get_gammapm(self):
        gammapm = np.exp(np.abs(self.zcat*self.zan)*(-self.A)*(np.sqrt(self.I)/(1+self.b*np.sqrt(self.I))
                         +2/self.b*np.log(1+self.b*np.sqrt(self.I)))
                         +self.m*2*self.vcat*self.van/self.v*self.Bgamma
                         +2*(self.vcat*self.van)**(3./2.)/self.v*self.Cgamma*self.m**2)
        return gammapm
