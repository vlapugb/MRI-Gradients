import scipy
import numpy as np
import pyfde
import math
from scipy import special
import random

mu_0 = 4*np.pi*10E-7 

def calc_grad(I, a, z_0, r, d, vitki):
    k = np.sqrt(4*a*r / ((d/2-z_0)**2 + (a+r)**2))
    K = scipy.special.ellipk(k**2)
    E = scipy.special.ellipe(k**2)
    dK_dk = E / (k*(1-k**2)) - K/k
    dE_dk = (E-K) / k
    P =(k**3*a + k*r*(k**2-2))/(2*(1-k**2))
    dP_dk = (k**2*(3*a+r)-k**4*(a+r)-2*r)/(2*(1-k**2)**2)
    dk_dz = -(d/2-z_0)*k**3/(4*a*r)
    return I*vitki*(mu_0 / 4 / np.pi/(r*np.sqrt(a*r))) * (r*K + r*k*dK_dk + E*dP_dk + P*dE_dk)*dk_dz
def B0(I, a, d, vitki):

    B = 3 * mu_0* vitki * I * (a ** 2) / 2 * ((d / 2) / ((d / 2) ** 2 + a ** 2) ** (5 / 2)
                                            + (d / 2) / ((d / 2) ** 2 + a ** 2) ** (5 / 2))
    return B
I = [1,2]
a = [3,4]
d = [0.2,0.3]
vitki = [5,6]
GradientSummary = 0
for j in range(30):
        r = random.uniform(0.01, 0.1)
        z_0 = random.uniform(-0.1,0.1)
        for i in range(2):
            GradientSummary += calc_grad(I[i],a[i], z_0, r, d[i], vitki[i]) - B0(I[i], a[i], d[i], vitki[i])
print(GradientSummary)