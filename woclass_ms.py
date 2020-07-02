import taichi as ti
import taichi_glsl as tl
import matplotlib.pyplot as plt
import numpy as np
ti.init()

dt = 0.1
N = 3
beta = 0.5
stiff = 1
gravity = 0
damping = 0

M = ti.var(ti.f32, (N, N))
J = ti.var(ti.f32, (N, N))
A = ti.var(ti.f32, (N, N))
rest = ti.var(ti.f32, (N, N))
F = ti.var(ti.f32, N)
b = ti.var(ti.f32, N)
x = ti.var(ti.f32, N)
v = ti.var(ti.f32, N)

@ti.func
def init_M():
    for i in range(N):
        M[i, i] = 1

@ti.func
def update_J():
    for i, d in J:
        for j in range(N):
            l_ij = rest[i, j]
            if l_ij != 0 and (d == i or d == j):
                x_ij = x[i] - x[j]
                J[i, d] += -stiff * (1 - l_ij / abs(x_ij))
                if d == j:
                    J[i, d] *= -1.0

@ti.func
def update_A():
    for i, j in A:
        A[i, j] = M[i, j] - beta * dt**2  * J[i, j]

@ti.func
def update_F():
    for i in range(N):
        F[i] = M[i, i] * gravity

    for i, j in rest:
        l_ij = rest[i, j]
        if l_ij != 0:
            x_ij = x[i] - x[j]
            F[i] += -stiff * (abs(x_ij) - l_ij) * tl.sign(x_ij)

@ti.func
def update_b():
    for i in range(N):
        v_star = v[i] * ti.exp(-dt * damping)
        b[i] = A[i, i] * v_star + dt * F[i]

@ti.func
def jacobi():
    for i in range(N):
        for j in range(N):
            if i != j:
                b[i] -= A[i, j] * v[j]

        v[i] = b[i] / A[i, i]

@ti.func
def implicit_euler():
    init_M()
    update_J()
    update_A()
    update_F()
    update_b()
    jacobi()

@ti.kernel
def substep():
    implicit_euler()
    for i in x:
        if x[i] < 0 and v[i] < 0 or x[i] > 1 and v[i] > 0:
            v[i] = -v[i]
    for i in x:
        x[i] += v[i] * dt

@ti.kernel
def init():
    for i in x:
        x[i] = ti.random()
    rest[0, 1] = 0.5
    rest[1, 0] = 0.5

init()
with ti.GUI('Implicit Euler') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        substep()
        gui.circles(x.to_numpy().reshape(N, 1) + np.zeros((N, 2)), radius=2)
        gui.show()
