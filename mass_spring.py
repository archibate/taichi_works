import taichi as ti
import taichi_glsl as tl
import numpy as np
import math
ti.init()


dt = 0.01
n = 16
N = n ** 2
W = 1
L = W / n
beta = 0.5
beta_dt = beta * dt
alpha_dt = (1 - beta) * dt
stiff = 100


x = ti.Vector(2, ti.f32, N)
v = ti.Vector(2, ti.f32, N)
b = ti.Vector(2, ti.f32, N)
F = ti.Vector(2, ti.f32, N)
K = ti.var(ti.f32)
ti.root.bitmasked(ti.ij, N).place(K)


@ti.func
def link(i, j, k):
    K[j, i] = k
    K[i, j] = k

@ti.kernel
def init():
    for i, j in ti.ndrange(n, n):
        a = i * n + j
        x[a] = ((tl.vec(i, j) + 0.5) * L - 0.5)
    for i, j in ti.ndrange(n - 1, n - 1):
        a = i * n + j
        link(a, a + n, stiff)
        link(a + n, a + n + 1, stiff)
        link(a + n + 1, a + 1, stiff)
        link(a + 1, a, stiff)

@ti.kernel
def explicit():
    '''
    v' = v + Mdt @ v
    x' = x + v'dt
    '''
    for i, j in K:
        disp = x[j] - x[i]
        d = disp.norm()
        acc = K[i, j] * disp * (d - L) / L ** 2
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

@ti.func
def Acc(v: ti.template(), x: ti.template(), dt):
    for i, j in K:
        disp = x[j] - x[i]
        acc = K[i, j] * disp * (disp.norm() - L) / L ** 2
        v[i] += acc * dt

@ti.kernel
def prepare():
    for i in x:
        x[i] += v[i] * alpha_dt
    Acc(v, x, alpha_dt)
    for i in x:
        b[i] = x[i]
        x[i] += v[i] * beta_dt

@ti.kernel
def jacobi():
    for i in v:
        b[i] = x[i] + F[i] * beta_dt ** 2
        F[i] = [0.0, 0.0]
    Acc(F, b, 1)

@ti.kernel
def collide():
    for i in x:
        v[i].y -= 0.01 * dt
    for i in x:
        v[i] = tl.boundReflect(b[i], v[i], -1, 1)

@ti.kernel
def update_pos():
    for i in x:
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

        x_ = x.to_numpy() * 0.5 + 0.5

        if 1:
            kar = np.empty((N * 2, 2), np.int32)
            kar_n = export_lines(kar)
            for i in range(kar_n):
                a, b = kar[i]
                gui.line(x_[a], x_[b])

        gui.circles(x_, radius=2)
        gui.show()
