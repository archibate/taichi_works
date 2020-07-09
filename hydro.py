import numpy as np
import taichi as ti
import taichi_glsl as tl
import matplotlib.cm as cm

cmap = cm.get_cmap('magma')


class Pair:
    def __init__(self, old, new):
        self.old = old
        self.new = new

    def swap(self):
        self.old, self.new = self.new, self.old


N = 600
NN = N, N
dx = 1 / N
dt = 0.01

dye = Pair(ti.var(ti.f32, NN), ti.var(ti.f32, NN))
pre = Pair(ti.var(ti.f32, NN), ti.var(ti.f32, NN))
vel = Pair(ti.Vector(2, ti.f32, NN), ti.Vector(2, ti.f32, NN))
div = ti.var(ti.f32, NN)


@ti.kernel
def advect(new: ti.template(), old: ti.template(), vel: ti.template()):
    for P in ti.grouped(ti.ndrange(*NN)):
        vel_P = vel[P]
        btP = P - vel_P * (N * dt)
        new[P] = tl.bilerp(old, btP)


@ti.kernel
def divergence(vel: ti.template()):
    for P in ti.grouped(ti.ndrange(*NN)):
        l = tl.sample(vel, P + tl.D.zy).x
        r = tl.sample(vel, P + tl.D.xy).x
        b = tl.sample(vel, P + tl.D.yz).y
        t = tl.sample(vel, P + tl.D.yx).y
        div[P] = (r - l + t - b) * (0.125 * dx)


@ti.kernel
def jacobi(new: ti.template(), old: ti.template()):
    for P in ti.grouped(ti.ndrange(*NN)):
        l = tl.sample(old, P + tl.D.zy)
        r = tl.sample(old, P + tl.D.xy)
        b = tl.sample(old, P + tl.D.yz)
        t = tl.sample(old, P + tl.D.yx)
        new[P] = (r + l + t + b) * 0.25 - div[P]


@ti.kernel
def sub_grad(vel: ti.template(), pre: ti.template()):
    for P in ti.grouped(ti.ndrange(*NN)):
        l = tl.sample(pre, P + tl.D.zy)
        r = tl.sample(pre, P + tl.D.xy)
        b = tl.sample(pre, P + tl.D.yz)
        t = tl.sample(pre, P + tl.D.yx)
        vel[P] = vel[P] - (0.5 / dx) * tl.vec(r - l, t - b)


def substep():
    advect(vel.new, vel.old, vel.old)
    advect(dye.new, dye.old, vel.old)
    dye.swap()
    vel.swap()

    divergence(vel.old)
    for _ in range(10):
        jacobi(pre.new, pre.old)
        pre.swap()

    sub_grad(vel.old, pre.old)


@ti.kernel
def drag(dye: ti.template(), vel: ti.template(),
            mx: ti.f32, my: ti.f32, dx: ti.f32, dy: ti.f32):
    mouse_size = ti.static(16)
    mouse_strength = ti.static(0.25)
    drag_strength = ti.static(36.0)

    dp = tl.vec(dx, dy) * drag_strength
    mP = int(tl.vec(mx, my) * N)
    for P in ti.grouped(ti.ndrange(*NN)):
        scale = tl.smoothstep(tl.distance(mP, P), mouse_size * 2, mouse_size)
        dye[P] = dye[P] + mouse_strength * scale
        vel[P] = vel[P] + mouse_strength * scale * dp


px, py = 0, 0

gui = ti.GUI('Quick-Fluid', NN)
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False

    mx, my = gui.get_cursor_pos()

    if gui.is_pressed(gui.LMB):
        dx, dy = mx - px, my - py
        drag(dye.old, vel.old, mx, my, dx, dy)

    px, py = mx, my

    substep()
    gui.set_image(dye.old)
    #gui.set_image(cmap(dye.old.to_numpy()))
    gui.show()
