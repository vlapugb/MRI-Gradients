import scipy
import numpy as np
import pyfde
import math
from scipy import special
import random

mu_0 = 1.26 * 10 ** (-6)

def calc_grad(I, a, z_0, r, d, vitki):
    k = np.sqrt(4*a*r / ((d/2-z_0)**2 + (a+r)**2))
    K = scipy.special.ellipk(k**2)
    E = scipy.special.ellipe(k**2)
    dK_dk = E / (k*(1-k**2)) - K/k
    dE_dk = (E-K) / k
    P =(k**3*a + k*r*(k**2-2))/(2*(1-k**2))
    dP_dk = (k**2*(3*a+r)-k**4*(a+r)-2*r)/(2*(1-k**2)**2)
    dk_dz = -(d/2-z_0)*k**3/(4*a*r)
    return 2*I*vitki/(r*np.sqrt(a*r)) * (r*K + r*k*dK_dk + E*dP_dk + P*dE_dk)*dk_dz

def B0(I, a, d, vitki):

    B = 3 * mu_0 * vitki * I * (a ** 2) / 2 * ((d / 2) / ((d / 2) ** 2 + a ** 2) ** (5 / 2)
                                            + (d / 2) / ((d / 2) ** 2 + a ** 2) ** (5 / 2))
    return B

print('Количество витков')
o = int(input())
print('Количетсво пар колец')
N = int(input()) # количество пар колец (вводим)
print(N)
k = 50 #точность измерений # mu0
rho = 0.0172 * 10 ** (-6) # удельное сопротивление
s = 0.0028 # поперечное сечение проволоки
parameters = np.zeros(4*N)  # точка старта оптимизаци и, по умолчанию заполнена 0
tau = 0.005 #время импульса МРТ ?
z = np.linspace(-0.1, 0.1, k) # массив точек для среднеквадратичного отклоенения
                             # (самая левая точка, самая правая точка, количество точек)

R = np.zeros(N) # массив сопротивлений
L = np.zeros(N) # массив индуктивностей

def err(parameters):
    I = parameters[:N] # массив токов
    a = parameters[N:2 * N] # массив радиусов a
    d = parameters[2 * N: 3 * N] # массив d
    vitki = parameters[3 * N: 4 * N]
    dBdy = np.zeros(N)  # для расчета 4 слагаемого
    sm2 = 0
    sm4 = 0
    GradientSummary = 0
    for j in range(300):
        r = random.uniform(0.01, 0.1)
        z_0 = random.uniform(-0.1,0.1)
        for i in range(2):
            GradientSummary += calc_grad(I[i],a[i], z_0, r, d[i], vitki[i]) - B0(I[i], a[i], d[i], vitki[i])



    for i in range(len(a)):
        L[0] = mu_0 * a[0] * vitki[0] ** 2 * math.log(abs(16 * a[0] / s) - 2) # формула для индуктивности
        L[1] = mu_0 * a[1] * vitki[1] ** 2 * math.log(abs(np.sqrt(8) * 16 * a[1] / s) - 2)
    for i in range(len(I)):
            for k in range(len(L)):
                sm2 =  sm2 + I[i] ** 2 * L[k] # Джоулевы потери


    for i in range(2):
        dBdy[i] = 3 * mu_0 * vitki[i] * I[i] * a[i] ** 2 / 2 * ((d[i] / 2) / ((d[i] / 2) ** 2 +
                            a[i] ** 2) ** (5 / 2) + (d[i] / 2) / ((d[i] / 2) ** 2 + a[i] ** 2) ** (5 / 2))

    sm3 = 0.2 ** 2 / (2 * mu_0) * sum(num ** 2 for num in dBdy)


    for i in range(N):
        R[0] = rho * 8 * vitki[0] * a[0] / (s ** 2)
        R[1] = rho * 8 * 8 * vitki[1] * a[1] / (s ** 2)

    for j in range(N):
        sm4 = sm4 + R[j] * (I[j] ** 2)

    myfunc = 0.2 ** 2 / (2 * mu_0) * GradientSummary / 300 *10**10  + sm2*10**4 + 2 * tau *  sm4 * 10**5 -  10**5 * abs(sm3)
    return myfunc 

#10 ** 2 * 0.2 ** 2 / (2 * mu_0) * sm1 / 50 + 10 ** 4 * sm2 + 2 * tau *10 ** 5 *  sm4 - 10 ** 5 * abs(sm3)
#sm1= -2.7902784192417954e-22
#sm2 = -0.00027984900292940948
#sm3 = -0.00002809856000000002
#sm4 = -4.0133333333333324e-05


limits = [(80, 80)]+[(8, 8)] + [(0.05, 0.13)] + [(0.1, 0.5)] + [(0.26, 0.5)]+ [(0.75, 0.80)] + [(10,10)] + [(10,10)]
#[(80, 80)] + [(0.15, 0.25)] + [(0.4, 0.6)] + [(o, o)]
#[(80, 80)]+[(10, 10)] + [(0.216, 0.216)] + [(0.22, 0.25)]  + [(0.4, 0.4)]+ [(0.45, 0.6)] + [(o,o)] + [(10,10)]
#[(80, 80)]+[(10, 10)] + [(0.179, 0.179)] + [(0.19, 0.25)]  + [(0.4, 0.4)]+ [(0.45, 0.6)] + [(o,o)] + [(10,10)]
solver = pyfde.JADE(fitness=err, n_dim=4*N, n_pop=400, limits=limits, minimize=True)
#solver.cr , solver.f = 0.5, 0.5
best, fit = solver.run(n_it=150)
sumL = 1/(sum(1/(2*num) for num in L))
sumR = 1/(sum(1/(2*num) for num in R))
print("Количество пар колец:", N)
print("Количество точек для СКО:", k)
print("Суммарное сопротивление", sumR)
print("Суммарная индуктивность", sumL)
print("Finish fitting:")
print("I(current):", best[0:N])
print("a(radius):", best[N:2*N])
print("d(distance btw coils):", best[2*N:3*N])
I1 = best[0:N]
a1 = best[N:2*N]
d1 = best[2*N:3*N]
vitki1 = best[3*N:4*N]

print(L)

print(R)
print("Количество витков:",best[3*N:4*N])
print("Эффективность", 1000 * sum([B0(I1[i], a1[i], d1[i], vitki1[i]) for i in range(N)]) / sum(i for i in I1))
print(fit)
