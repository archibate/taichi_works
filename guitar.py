import taichi as ti
import taichi_glsl as tl
ti.init()


dt = 0.03
N = 32


x = ti.var(ti.f32, N)
v = ti.var(ti.f32, N)
A = ti.var(ti.f32, (N, N))
K = ti.var(ti.f32)
ti.root.bitmasked(ti.ij, N).place(K)


@ti.kernel
def init():
    for i in range(1, N):
        K[i - 1, i] = 1
        K[i, i - 1] = 1

@ti.kernel
def explicit():
    '''
    v' = v + Mdt @ v
    x' = x + v'dt
    '''
    for i, j in K:
        v[i] += dt * K[i, j] * (x[j] - x[i])
    for i in x:
        x[i] += dt * v[i]

#############################################
'''
v' = v + Mdt @ v'
(I - Mdt) @ v' = v
'''

@ti.kernel
def init_A():
    '''
    #dv sx
    M[i, j] = K[i, j]
    M[i, i] = -sum(K[i, j] for j)
    A[i, j] = -M[i, j] * dt
    '''
    for i, j in A:
        A[i, j] = abs(i == j)
    for i, j in K:
        A[i, j] = -K[i, j] * dt
        A[i, i] += K[i, j] * dt

@ti.kernel
def step_A():
    for i in v:
        for j in range(N):
            v[i] += -A[i, j] * x[j]
    for i in x:
        x[i] += v[i] * dt

@ti.kernel
def update_x():
    for i in x:
        x[i] += dt * v[i]


def implicit():
    init_A()
    for i in range(100):
        step_A()
    update_x()


init()
with ti.GUI('Guitar', (1024, 256)) as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if gui.is_pressed(gui.LMB) or gui.is_pressed(gui.RMB):
            mx, my = gui.get_cursor_pos()
            i = int(mx * N)
            gui.rect((i / N, 0), ((i + 1) / N, 1), color=0x666666)
            if gui.is_pressed(gui.LMB):
                x[i] = my * 2 - 1
            else:
                v[i] = my * 2 - 1
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
