import taichi as ti
import taichi_glsl as tl
import numpy as np
ti.init()


dt = 0.07
N = 3
beta = 0.3
beta_dt = beta * dt
alpha_dt = (1 - beta) * dt


x = ti.Vector(2, ti.f32, N)
v = ti.Vector(2, ti.f32, N)
b = ti.Vector(2, ti.f32, N)
F = ti.Vector(2, ti.f32, N)
K = ti.var(ti.f32)
A = ti.var(ti.f32)
ti.root.bitmasked(ti.ij, N).place(K)
ti.root.bitmasked(ti.ij, N).place(A)


@ti.func
def link(i, j, k):
    K[j, i] = k
    K[i, j] = k

@ti.kernel
def init():
    link(0, 1, 1.0)
    link(0, 2, 1.0)
    link(1, 2, 1.0)
    ang = 0.0
    x[0] = [ti.sin(ang), ti.cos(ang)]
    ang += ti.math.tau / 3
    x[1] = [ti.sin(ang), ti.cos(ang)]
    ang += ti.math.tau / 3
    x[2] = [ti.sin(ang), ti.cos(ang)]

@ti.kernel
def explicit():
    '''
    v' = v + Mdt @ v
    x' = x + v'dt
    '''
    for i, j in K:
        acc = K[i, j] * (x[j] - x[i])
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

@ti.kernel
def init_A():
    '''
    #dv sx
    M[i, j] = K[i, j]
    M[i, i] = -sum(K[i, j] for j)
    A[i, j] = M[i, j] * dt
    '''
    for i, j in K:
        A[i, j] = K[i, j]
        A[i, i] -= K[i, j]

@ti.kernel
def prepare():
    for i in x:
        x[i] += v[i] * alpha_dt
    for i, j in A:
        v[i] += A[i, j] * x[j] * alpha_dt
    for i in x:
        b[i] = x[i]
        x[i] += v[i] * beta_dt

@ti.kernel
def jacobi():
    for i in v:
        b[i] = x[i] + F[i] * beta_dt ** 2
        F[i] = 0.0
    for i, j in A:
        F[i] += A[i, j] * b[j]

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

@ti.kernel
def export_lines(out: ti.ext_arr()) -> ti.i32:
    n = 0
    for i, j in K:
        print(i, j)
        if i < j:
            p = ti.atomic_add(n, 1)
            out[p, 0] = i
            out[p, 1] = j
    return n


init()
init_A()
with ti.GUI('Mass Spring') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if gui.is_pressed(gui.LMB) or gui.is_pressed(gui.RMB):
            mx, my = gui.get_cursor_pos()
            i = int(mx * N)
            gui.rect((i / N, 0), ((i + 1) / N, 1), color=0x666666)
            if gui.is_pressed(gui.LMB):
                x[i].y = (my * 2 - 1) * 3
            else:
                v[i].y = (my * 2 - 1) * 3
        if gui.is_pressed('r'):
            x.fill(0)
            v.fill(0)

        if not gui.is_pressed(gui.SPACE):
            explicit()

        x_ = x.to_numpy() * 0.5 + 0.5

        kar = np.empty((N * 2, 2), np.int32)
        kar_n = export_lines(kar)
        for i in range(kar_n):
            a, b = kar[i]
            gui.line(x_[a], x_[b])

        gui.circles(x_, radius=2)
        gui.show()
