import taichi as ti
import taichi_glsl as tl
ti.init()


dt = 0.07
N = 256
beta = 0.3
beta_dt = beta * dt
alpha_dt = (1 - beta) * dt


x = ti.var(ti.f32, N)
v = ti.var(ti.f32, N)
b = ti.var(ti.f32, N)
F = ti.var(ti.f32, N)
D = ti.var(ti.f32, N)
K = ti.var(ti.f32)
A = ti.var(ti.f32)
ti.root.bitmasked(ti.ij, N).place(K)
ti.root.bitmasked(ti.ij, N).place(A)


@ti.kernel
def init():
    for i in range(1, N):
        K[i - 1, i] = 100
        K[i, i - 1] = 100
    for i in range(N):
        D[i] = 1

@ti.kernel
def explicit():
    '''
    v' = v + Mdt @ v
    x' = x + v'dt
    '''
    for i, j in K:
        acc = K[i, j] * (x[j] - x[i]) - D[i] * x[i]
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
    M[i, i] = -sum(K[i, j] for j) - D[i]
    A[i, j] = M[i, j] * dt
    '''
    for i, j in K:
        A[i, j] = K[i, j]
        A[i, i] -= K[i, j] + D[i]

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


init()
init_A()
with ti.GUI('Guitar', (1024, 256)) as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if gui.is_pressed(gui.LMB) or gui.is_pressed(gui.RMB):
            mx, my = gui.get_cursor_pos()
            i = int(mx * N)
            gui.rect((i / N, 0), ((i + 1) / N, 1), color=0x666666)
            if gui.is_pressed(gui.LMB):
                x[i] = (my * 2 - 1) * 3
            else:
                v[i] = (my * 2 - 1) * 3
        if gui.is_pressed('r'):
            x.fill(0)
            v.fill(0)

        if not gui.is_pressed(gui.SPACE):
            implicit()
        pp = None
        for i in range(N):
            cp = ((i + 0.5) / N, x[i] * 0.5 + 0.5)
            gui.circle(cp, radius=2)
            if pp is not None:
                gui.line(pp, cp, radius=1)
            pp = cp
        gui.show()
