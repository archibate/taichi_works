import taichi as ti
import taichi_glsl as tl
import matplotlib.cm as cm
cmap = cm.get_cmap('magma')


x = ti.var(ti.f32, (512, 512))
y = ti.var(ti.f32, (512, 512))
b = ti.var(ti.i32, (512, 512))


links = [ti.Vector(_) for _ in [[-1, 0], [1, 0], [0, -1], [0, 1]]]
@ti.kernel
def possion(x: ti.template(), y: ti.template()):
    for i in ti.grouped(x):
        if b[i] == 1:
            continue
        avg = x[i] * 0
        for dir in ti.static(links):
            avg += x[i + dir]
        y[i] = avg / len(links)


@ti.kernel
def init():
    for i in ti.grouped(x):
        x[i] = 0.0
    for i in range(512):
        b[i, 0] = 1
        b[i, 511] = 1
        b[511, i] = 1
        b[0, i] = 1


@ti.kernel
def press_x(mx: ti.f32, my: ti.f32):
    m = tl.vec(mx, my)
    mi = int(m * 512)
    for i in ti.grouped(x):
        x[i] = max(x[i], 2 * ti.exp(-0.05 * (i - mi).norm_sqr()))


@ti.kernel
def press_b(mx: ti.f32, my: ti.f32):
    m = tl.vec(mx, my)
    mi = int(m * 512)
    for i in ti.grouped(x):
        s = tl.step(tl.sqrLength(float(i - mi)), 10)
        b[i] = max(b[i], s)
        x[i] = max(x[i], s)


init()
with ti.GUI('Possion Solver') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if gui.is_pressed(gui.LMB):
            press_b(*gui.get_cursor_pos())
        if gui.is_pressed(gui.RMB):
            press_x(*gui.get_cursor_pos())
        for i in range(10):
            possion(x, y)
            possion(y, x)
        gui.set_image(cmap(x.to_numpy()))
        gui.show()
