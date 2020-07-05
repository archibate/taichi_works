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

N = 128
M = 32
dt = 0.00001
steps = 10
cmap = cm.get_cmap('magma')
p_vor = ti.var(ti.f32, N)
p_pos = ti.Vector(2, ti.f32, N)
g_com = ti.Vector(2, ti.f32, (M, M))
g_vo0 = ti.var(ti.f32, (M, M))
g_vo1 = ti.Vector(2, ti.f32, (M, M))
g_cnt = ti.var(ti.i32, (M, M))
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
    #for g in ti.grouped(g_com):
    for g in ti.grouped(ti.ndrange(M, M)):
        vel += compute(p - g_com[g], g_vo0[g], g_vo1[g])
    return vel


@ti.kernel
def advance():
    for i in p_pos:
        vel = velocity(p_pos[i])
        p_pos[i] = p_pos[i] + vel * dt


@ti.kernel
def advance_fmm():
    build_fmm()
    for i in p_pos:
        vel = velocity_fmm(p_pos[i])
        p_pos[i] = p_pos[i] + vel * dt


@ti.kernel
def init():
    p_pos[0] = vec(0.489, 0.49)
    p_pos[1] = vec(0.491, 0.49)
    p_vor[0] = +500.0
    p_vor[1] = -500.0


@ti.kernel
def init():
    for i in range(N):
        p_pos[i] = tl.randND(2)
        p_vor[i] = tl.randRange(-1.0, 1.0)


@ti.func
def build_fmm():
    for g in ti.grouped(g_com):
        g_vo0[g] = 0.0
        g_vo1[g] = vec2(0.0)
        g_com[g] = vec2(0.0)
        g_cnt[g] = 0
    for i in p_pos:
        g = int(p_pos[i] * M)
        g_com[g] += p_pos[i]
        g_vo0[g] += p_vor[i]
        g_cnt[g] += 1
    for g in ti.grouped(g_com):
        if g_cnt[g] != 0:
            g_com[g] = g_com[g] / g_cnt[g]
        else:
            g_com[g] = (g + 0.5) / M
    for i in p_pos:
        g = int(p_pos[i] * M)
        g_vo1[g] += p_vor[i] * (g_com[g] - p_pos[i]).Yx


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
            advance_fmm()
        img.fill(0.0)
        render(*gui.get_cursor_pos())
        colors = rgb_to_hex(cmap(np.abs(p_vor.to_numpy())).transpose())
        gui.set_image(img)
        gui.circles(p_pos.to_numpy(), radius=2, color=colors)
        gui.show()
        gui.frame += 1
