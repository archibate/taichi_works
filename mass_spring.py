import taichi as ti
import taichi_glsl as tl
import taichi_three as t3
import numpy as np
import math
ti.init()


dt = 0.003
N = 32
NN = N, N
W = 1
L = W / N
beta = 0.5
beta_dt = beta * dt
alpha_dt = (1 - beta) * dt
stiff = 2000


x = ti.Vector(2, ti.f32, NN)
v = ti.Vector(2, ti.f32, NN)
b = ti.Vector(2, ti.f32, NN)
F = ti.Vector(2, ti.f32, NN)


@ti.kernel
def init():
    for i in ti.grouped(ti.ndrange(N, N)):
        x[i] = (i + 0.5) * L - 0.5

@ti.kernel
def explicit():
    '''
    v' = v + Mdt @ v
    x' = x + v'dt
    '''
    for i, j in K:
        disp = x[j] - x[i]
        acc = K[i, j] * disp * (disp.norm() - L) / L ** 2
        v[i] += dt * acc
    for i in x:
        x[i] += dt * v[i]

#############################################
'''
v' = v + Mdt @ v'
(I - Mdt) @ v' = v
'''

'''
v' = v + Mdt @ [beta v' + alpha v]
(I - beta Mdt) @ v' = (I + alpha Mdt) @ v
'''

links = [tl.vec(*_) for _ in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

@ti.func
def Acc(v: ti.template(), x: ti.template(), dt):
    for i in ti.grouped(x):
        acc = tl.vec2(0.0)
        for d in ti.static(links):
            disp = x[tl.clamp(i + d, 0, tl.vec(*NN) - 1)] - x[i]
            acc += disp * (disp.norm() - L) / L ** 2
        v[i] += stiff * acc * dt

@ti.kernel
def prepare():
    for i in ti.grouped(x):
        x[i] += v[i] * alpha_dt
    Acc(v, x, alpha_dt)
    for i in ti.grouped(x):
        b[i] = x[i]
        x[i] += v[i] * beta_dt

@ti.kernel
def jacobi():
    for i in ti.grouped(x):
        b[i] = x[i] + F[i] * beta_dt ** 2
        F[i] = tl.vec2(0)
    Acc(F, b, 1)

@ti.kernel
def collide():
    for i in ti.grouped(x):
        v[i].y -= 0.98 * dt
    for i in ti.grouped(x):
        v[i] = tl.boundReflect(b[i], v[i], -1, 1)

@ti.kernel
def update_pos():
    for i in ti.grouped(x):
        x[i] = b[i]
        v[i] += F[i] * beta_dt

def implicit():
    prepare()
    for i in range(12):
        jacobi()
    update_pos()
    collide()

@ti.kernel
def export_lines(out: ti.ext_arr()) -> ti.i32:
    n = 0
    for i, j in K:
        if i < j:
            p = ti.atomic_add(n, 1)
            out[p, 0] = i
            out[p, 1] = j
    return n


init()
with ti.GUI('Mass Spring') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if not gui.is_pressed(gui.SPACE):
            implicit()

        x_ = x.to_numpy().reshape(N * N, 2) * 0.5 + 0.5

        if 0:
            kar = np.empty((N * 2, 2), np.int32)
            kar_n = export_lines(kar)
            for i in range(kar_n):
                a, b = kar[i]
                gui.line(x_[a], x_[b])

        gui.circles(x_, radius=2)
        gui.show()
