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
def tree_append(pos):
    i = abs(pos >= 0.5).dot(vec(1, 2)) + 1
    #while ti.is_active(tree, i):
    nodes[i] = pos

@ti.kernel
def build_tree():
    for i in particles:
        tree_append(particles[i])

@ti.func
def paint_tree():
    for i in ti.static(range(1, 4 + 1)):
        par = vec(i % 2, i // 2) * 256
        rap = par
        if ti.is_active(tree, i):
            rap = par + 256
        for j, k in ti.ndrange((par.x, rap.x), (par.y, rap.y)):
            image[j, k] = 0.1


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
