## Initialization
import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec, vec2, vec3, math

## Classes
def Vector2(*args, **kwargs):
    return ti.Vector(2, ti.f32, *args, **kwargs)


## Define Variables
N = 2
L = 4
particles = Vector2(N)
image = ti.var(ti.f32, (512, 512))

nodes = Vector2()
tree = ti.root.pointer(ti.i, 4 ** (L + 1))
tree.place(nodes)


## Algorithms
@ti.func
def tree_append(p):
    i = int(p * 2).dot(vec(1, 2)) + 4
    if ti.is_active(tree, i):
        q = nodes[i]
        nodes[i] = (p + q) / 2
        i = int(p * 4).dot(vec(1, 4)) + 16
        j = int(q * 4).dot(vec(1, 4)) + 16
        nodes[j] = q
    nodes[i] = p

@ti.kernel
def build_tree():
    for i in particles:
        tree_append(particles[i])

@ti.func
def paint_tree():
    for i in ti.static(range(4)):
        par = vec(i % 2, i // 2) * 256
        rap = par
        if ti.is_active(tree, i + 4):
            rap = par + 256
        for j, k in ti.ndrange((par.x, rap.x), (par.y, rap.y)):
            image[j, k] = max(image[j, k], 0.1)
    for i in range(16):
        par = vec(i % 4, i // 4) * 128
        rap = par
        if ti.is_active(tree, i + 16):
            print(i)
            rap = par + 128
        for j, k in ti.ndrange((par.x, rap.x), (par.y, rap.y)):
            image[j, k] = max(image[j, k], 0.2)
    for i in range(64):
        par = vec(i % 8, i // 8) * 64
        rap = par
        if ti.is_active(tree, i + 64):
            print(i)
            rap = par + 128
        for j, k in ti.ndrange((par.x, rap.x), (par.y, rap.y)):
            image[j, k] = max(image[j, k], 0.2)


## Helper Functions
@ti.func
def npow(x):
    d = x * 0
    if any(x != 0):
        d = ts.normalizePow(x, -2, 1e-3)
    return d


## Main Program
@ti.kernel
def init():
    for i in particles:
        particles[i] = ts.randND(2)

@ti.func
def compute_grad(p):
    acc = p * 0
    for i in particles:
        acc += npow(particles[i] - p)
    return acc * 0.001

@ti.kernel
def render(mx: ti.f32, my: ti.f32):
    p = vec(mx, my)
    acc = compute_grad(p)
    tree_append(p)
    ts.paintArrow(image, p, acc)
    paint_tree()


## GUI Loop
init()
with ti.GUI('FFM Gravity') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        image.fill(0)
        tree.deactivate_all()
        build_tree()
        render(*gui.get_cursor_pos())
        gui.set_image(image)
        gui.circles(particles.to_numpy(), radius=2)
        gui.show()
