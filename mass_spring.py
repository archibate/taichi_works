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
stiff = 3000


x = ti.Vector(3, ti.f32, NN)
v = ti.Vector(3, ti.f32, NN)
b = ti.Vector(3, ti.f32, NN)
F = ti.Vector(3, ti.f32, NN)


@ti.kernel
def init():
    for i in ti.grouped(x):
        x[i] = tl.vec((i + 0.5) * L - 0.5, 0)

@ti.kernel
def explicit():
    '''
    v' = v + Mdt @ v
    x' = x + v'dt
    '''
    for i, j in K:
        disp = x[j] - x[i]
        acc = K[i, j] * disp * (disp.norm() - L) / L**2
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
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[tl.clamp(i + d, 0, tl.vec(*NN) - 1)] - x[i]
            acc += disp * (disp.norm() - L) / L**2
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
        b[i] = x[i] + F[i] * beta_dt**2
        F[i] = b[i] * 0
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


scene = t3.Scene()
model = t3.Model()
scene.add_model(model)

faces = t3.Face.var(N**2 * 3)
lines = t3.Line.var(N**2 * 2)
vertices = t3.Vertex.var(N**2)
model.set_vertices(vertices)
model.add_geometry(faces)
model.add_geometry(lines)

@ti.kernel
def init_display():
    for i_ in ti.grouped(ti.ndrange(N - 1, N - 1)):
        i = i_
        a = i.dot(tl.vec(N, 1))
        i.x += 1
        b = i.dot(tl.vec(N, 1))
        i.y += 1
        c = i.dot(tl.vec(N, 1))
        i.x -= 1
        d = i.dot(tl.vec(N, 1))
        i.y -= 1
        faces[a * 2 + 0].idx = tl.vec(a, b, c)
        faces[a * 2 + 1].idx = tl.vec(a, c, d)
        lines[a * 2 + 0].idx = tl.vec(a, b)
        lines[a * 2 + 1].idx = tl.vec(a, d)

@ti.kernel
def update_display():
    for i in ti.grouped(x):
        vertices[i.dot(tl.vec(N, 1))].pos = x[i]


init()
init_display()
scene.set_light_dir([-1, 1, -1])
with ti.GUI('Mass Spring') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if not gui.is_pressed(gui.SPACE):
            for i in range(1):
                implicit()
            update_display()

        scene.camera.from_mouse(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()
