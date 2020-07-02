import taichi as ti
import taichi_glsl as tl
ti.init()


dt = 0.05
N = 256
beta = 0.5
alpha = 1 - beta


x = ti.var(ti.f32, N)
y = ti.var(ti.f32, N)
v = ti.var(ti.f32, N)
u = ti.var(ti.f32, N)
A = ti.var(ti.f32, (N, N))
D = ti.var(ti.f32, N)
K = ti.var(ti.f32)
ti.root.bitmasked(ti.ij, N).place(K)


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
    for i, j in A:
        A[i, j] = 0
    for i, j in K:
        A[i, j] = K[i, j]
        A[i, i] -= K[i, j] + D[i]
    for i in x:
        x[i] = v[i] * alpha * dt + x[i]
        for j in range(N):
            v[i] += A[i, j] * x[j] * alpha * dt
    for i in x:
        y[i] = x[i]
        u[i] = v[i]

@ti.kernel
def jacobi_A():
    for i in v:
        u[i] = v[i]
        for j in range(N):
            u[i] += A[i, j] * y[j] * beta * dt
        y[i] = u[i] * beta * dt + x[i]

@ti.kernel
def update_x():
    for i in x:
        x[i] = y[i]
        v[i] = u[i]


def implicit():
    init_A()
    for i in range(12):
        jacobi_A()
    update_x()


init()
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
