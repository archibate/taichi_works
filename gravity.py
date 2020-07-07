import taichi as ti
import taichi_glsl as tl
import matplotlib.cm as cm
cmap = cm.get_cmap('magma')


N = 512

x = ti.var(ti.f32)
y = ti.var(ti.f32)
b = ti.var(ti.i32)

ti.root.dense(ti.ij, N).place(x)
ti.root.dense(ti.ij, N).place(y)
ti.root.dense(ti.ij, N).place(b)


links = [[-1, 0], [1, 0], [0, -1], [0, 1]]
weights = [1, 1, 1, 1]
links = [ti.Vector(_) for _ in links]
@ti.kernel
def possion(x: ti.template(), y: ti.template(), m: ti.template()):
    for i in ti.grouped(x):
        if b[i] != 0:
            continue
        avg = x[i] * 0
        for dir, wei in ti.static(zip(links, weights)):
            avg += wei * tl.sample(x, i + dir * m)
        y[i] = avg / sum(weights)


@ti.kernel
def init():
    for i in ti.grouped(x):
        x[i] = 0.0
    for i in range(N):
        b[i, 0] = 1
        b[i, N - 1] = 1
        b[N - 1, i] = 1
        b[0, i] = 1


@ti.kernel
def press_x(mx: ti.f32, my: ti.f32):
    m = tl.vec(mx, my)
    mi = int(m * N)
    for i in ti.grouped(x):
        x[i] = max(x[i], 2 * ti.exp(-0.05 * (i - mi).norm_sqr()))


@ti.kernel
def press_b(mx: ti.f32, my: ti.f32):
    m = tl.vec(mx, my)
    mi = int(m * N)
    b[mi] = max(b[mi], 1)
    x[mi] = max(x[mi], 1)


init()
with ti.GUI('Possion Solver') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if gui.is_pressed(gui.LMB):
            press_b(*gui.get_cursor_pos())
        if gui.is_pressed(gui.RMB):
            press_x(*gui.get_cursor_pos())
        for i in range(15):
            possion(x, y, 1)
            possion(y, x, 1)
        gui.set_image(cmap(x.to_numpy()))
        gui.show()
