# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:24:21 2019

@author: João Angelo Ferres Brogin
"""

import numpy as np
import picos as pic
import cvxopt as cvx
# Please change the name of 'Applying orthogonal polynomials for identifying state-space systems from real brain activity' to 'ID_seizures_CHUV_OP'
from ID_seizures_CHUV_OP import Xsol, Xsol_aux, A_est_h, A_est_s, X0_est_h, X0_est_s

# %% States and matrices of the system:
states_h = Xsol
states_s = Xsol_aux

Ah = A_est_h
As = A_est_s
Ag = A_est_h # We want to control a seizure based on the healthy model

Z_new = np.zeros((len(Ah)//2, len(Ah)//2))
I_new = np.identity(len(Ah)//2)
Bh = np.vstack((Z_new, I_new))
Bs = np.vstack((Z_new, I_new))
Bg = np.vstack((Z_new, I_new))

# Bh = B_est_h
# Bs = B_est_s
# Bg = Bh

X0h = X0_est_h
X0s = X0_est_s
X0g = X0_est_h

N = len(Ah)
alpha_h = 100
alpha_s = 50 # 50 base
alpha_g = 70 # 100 simple, 10 adaptive

C = np.ones((1, N))
I = np.identity(N//2)
mu = 0.4e6

# %% Optimization problem:
prob = pic.Problem()

# Track non-seizure activity:
AAh = pic.new_param('Ah', cvx.matrix(Ah))
BBh = pic.new_param('Bh', cvx.matrix(Bh))

Gxh = prob.add_variable('Gxh', (N//2, N))
Xh = prob.add_variable('Xh', (N, N), vtype='symmetric')

# Track seizure:
AAs = pic.new_param('As', cvx.matrix(As))
BBs = pic.new_param('Bs', cvx.matrix(Bs))

Gxs = prob.add_variable('Gxs', (N//2, N))
Xs = prob.add_variable('Xs', (N, N), vtype='symmetric')

# Control seizure:
AAg = pic.new_param('Ag', cvx.matrix(Ah))
BBg = pic.new_param('Bg', cvx.matrix(Bh))
X_0 = pic.new_param('X_0', cvx.matrix(X0g))
CC = pic.new_param('C', cvx.matrix(C))

Gxg = prob.add_variable('Gxg', (N//2, N))
Xg = prob.add_variable('Xg', (N, N), vtype='symmetric')

prob.add_constraint(Xh * AAh.T - Gxh.T * BBh.T + AAh * Xh - BBh * Gxh + 2 * alpha_h * Xh << 0)
prob.add_constraint(Xs * AAs.T - Gxs.T * BBs.T + AAs * Xs - BBs * Gxs + 2 * alpha_s * Xs << 0)
prob.add_constraint(Xg * AAg.T - Gxg.T * BBg.T + AAg * Xg - BBg * Gxg + 2 * alpha_g * Xg << 0)

# Constraints of the input:
prob.add_constraint( ( (1 & X_0.T )//(X_0 & Xg) ) >> 0 )
prob.add_constraint( ( (Xg & Gxg.T )//(Gxg & (mu**2)*I) ) >> 0 )

# Positiveness:
prob.add_constraint(Xh >> 0)
prob.add_constraint(Xs >> 0)
prob.add_constraint(Xg >> 0)

# Solver:
prob.solve(verbose=1)
print('Status: ' + prob.status)

# Health:
Xh = np.matrix(Xh.value)
Gxh = np.matrix(Gxh.value)

Ph = np.matrix(Xh).I
Gh = np.matrix(Gxh).dot(Ph)

# Seizure:
Xs = np.matrix(Xs.value)
Gxs = np.matrix(Gxs.value)

Ps = np.matrix(Xs).I
Gs = np.matrix(Gxs).dot(Ps)

# Control seizure:
Xg = np.matrix(Xg.value)
Gxg = np.matrix(Gxg.value)

Pg = np.matrix(Xg).I
Gg = np.matrix(Gxg).dot(Pg)
