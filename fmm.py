import numpy as np
import taichi as ti
import taichi_glsl as tl
import matplotlib.cm as cm
from taichi_glsl import vec, vec2, D
ti.init()

ti.Matrix.Yx = property(lambda u: vec(-u.y, u.x))

def rgb_to_hex(c):
    to255 = lambda x: np.minimum(255, np.maximum(0, np.int32(x * 255)))
    return 65536 * to255(c[0]) + 256 * to255(c[1]) + to255(c[2])

N = 2
M = 3
dt = 0.00001
steps = 0
cmap = cm.get_cmap('magma')
p_vor = ti.var(ti.f32, N)
p_pos = ti.Vector(2, ti.f32, N)
g_com = ti.Vector(2, ti.f32, M)
g_vo0 = ti.var(ti.f32, M)
g_vo1 = ti.Vector(2, ti.f32, M)
img = ti.Vector(3, ti.f32, (512, 512))
eps = 1e-5


@ti.func
def cmul(a, b):
    return vec(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

@ti.func
def crcp(a):
    return tl.normalizePow(a, -1, eps).Yx

@ti.func
def compute(d, m0, m1):
    d = crcp(d)
    r = d * m0
    r += cmul(cmul(d, d), m1)
    return r


@ti.func
def velocity(p):
    vel = vec2(0.0)
    for j in range(N):
        vel += compute(p - p_pos[j], p_vor[j], vec2(0.0))
    return vel


@ti.func
def velocity_fmm(p):
    vel = vec2(0.0)
    for j in range(N):
        vel += compute(p - p_pos[j], g_vo0[j], g_vo1[j])
    return vel


@ti.kernel
def init():
    p_pos[0] = vec(0.499, 0.50)
    p_pos[1] = vec(0.501, 0.50)
    p_vor[0] = +500.0
    p_vor[1] = -500.0


@ti.func
def build_fmm():
    g_com[0] = calc_com()
    g_vo0[0] = calc_m0(g_com[0])
    g_vo1[0] = calc_m1(g_com[0])


@ti.func
def calc_m1(g_com):
    mu1 = vec2(0.0)
    for i in p_pos:
        mu1 += -0.5 * p_vor[i] * crcp(p_pos[i]).Yx
    return mu1


@ti.func
def calc_m0(g_com):
    mu0 = 0.0
    for i in p_pos:
        mu0 += p_vor[i]
    return mu0


@ti.func
def calc_com():
    cen = vec2(0.0)
    tot = 0
    for i in p_pos:
        cen += p_pos[i]
        tot += p_vor[i]
    return cen / tot


@ti.kernel
def render(mx: ti.f32, my: ti.f32):
    mouse = vec(mx, my)

    dir = velocity(mouse) * 0.001
    if dir.norm() > 1:
        dir = dir.normalized()
    tl.paintArrow(img, mouse, dir, D.xyy)

    build_fmm()
    dir = velocity_fmm(mouse) * 0.001
    if dir.norm() > 1:
        dir = dir.normalized()
    tl.paintArrow(img, mouse, dir, D.yyx)



init()
with ti.GUI('Vortices', background_color=rgb_to_hex(cmap(0))) as gui:
    gui.frame = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        for i in range(steps):
            advance()
        img.fill(0.0)
        render(*gui.get_cursor_pos())
        colors = rgb_to_hex(cmap(np.abs(p_vor.to_numpy())).transpose())
        gui.set_image(img)
        gui.circles(p_pos.to_numpy(), radius=2, color=colors)
        gui.show()
        gui.frame += 1
