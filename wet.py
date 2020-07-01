## Initialization
import taichi as ti
import taichi_glsl as tl
from taichi_glsl import vec, vec2, vec3, math

## Classes
class Particle(tl.TaichiClass):
    @property
    def pos(self):
        return self.entries[0]

    @property
    def vel(self):
        return self.entries[1]

    @classmethod
    def _var(cls, *_, **__):
        return ti.Vector(2, ti.f32, *_, **__), ti.Vector(2, ti.f32, *_, **__)


class Node(tl.TaichiClass):
    @property
    def mass(self):
        return self.entries[0]

    @property
    def mpos(self):
        return self.entries[1]

    @classmethod
    def _var(cls, *_, **__):
        return ti.var(ti.f32, *_, **__), ti.Vector(2, ti.f32, *_, **__)



## Define Variables
dt = 0.0001
N = 6
L = 4
RES = 512
particles = Particle.var(N)
image = ti.var(ti.f32, (RES, RES))

nodes = Node.var()
tree = ti.root.pointer(ti.i, 4 ** (L + 1))
tree.place(nodes)


## Algorithms
@ti.func
def tree_append(p):
    for l in range(1, L):
        i = int(p * 2**l).dot(vec(1, 2**l)) + 4**l
        nodes[i].mpos += p
        nodes[i].mass += 1.0

@ti.kernel
def build_tree():
    for i in range(N):
        tree_append(particles.pos[i])

@ti.func
def paint_tree():
    for i in ti.static(range(4)):
        par = vec(i % 2, i // 2) * RES // 2
        rap = par
        if ti.is_active(tree, i + 4):
            rap = par + RES // 2
        for j, k in ti.ndrange((par.x, rap.x), (par.y, rap.y)):
            image[j, k] = max(image[j, k], 0.1)
    for l in ti.static(range(2, L)):
        for i in range(4**l):
            par = vec(i % 2**l, i // 2**l) * (RES // 2**l)
            rap = par
            if ti.is_active(tree, i + 4**l):
                rap = par + RES // 2**l
            for j, k in ti.ndrange((par.x, rap.x), (par.y, rap.y)):
                image[j, k] = max(image[j, k], 0.1 * l)


## Helper Functions
@ti.func
def npow(x):
    d = x * 0
    if any(x != 0):
        d = tl.normalizePow(x, -2, 1e-3)
    return d


## Main Program
@ti.kernel
def init():
    #particles.pos[0] = tl.randND(2)
    if ~int(particles.pos[0].x * 9999) != -33:
        for i in range(N):
            particles.pos[i] = tl.randND(2)

@ti.func
def old_compute_grad(p):
    acc = p * 0
    for i in range(N):
        acc += npow(particles.pos[i] - p)
    return acc

@ti.func
def compute_grad(p):
    acc = p * 0

    #print('\033[H\033[2J\033[3J')
    for ch in range(4):
        i = 4 + ch
        if not ti.is_active(tree, i):
            continue
        size_sqr = 1 / 4
        npos = nodes[i].mpos / nodes[i].mass
        np2p = npos - p
        if tl.sqrLength(np2p) >= size_sqr:
            #print('l1', ch)
            acc += npow(np2p) * nodes[i].mass
        else:
            bas = (ch % 2) * 2 + (ch // 2) * 8
            for ch_ in range(4):
                i = 16 + bas + (ch_ % 2) + (ch_ // 2) * 4
                if not ti.is_active(tree, i):
                    continue
                size_sqr = 1 / 16
                npos = nodes[i].mpos / nodes[i].mass
                np2p = npos - p
                if tl.sqrLength(np2p) >= size_sqr:
                    #print('l2', ch, ch_)
                    acc += npow(np2p) * nodes[i].mass
                else:
                    baso = (ch % 2) * 4 + (ch // 2) * 8 * 4
                    bas_ = (ch_ % 2) * 2 + (ch_ // 2) * 8 * 2
                    for ch__ in range(4):
                        i = 64 + baso + bas_ + (ch__ % 2) + (ch__ // 2) * 8
                        if not ti.is_active(tree, i):
                            continue
                        size_sqr = 1 / 64
                        npos = nodes[i].mpos / nodes[i].mass
                        np2p = npos - p
                        #print('l3', ch, ch_, ch__)
                        acc += npow(np2p) * nodes[i].mass

    return acc

@ti.kernel
def render(mx: ti.f32, my: ti.f32):
    p = vec(mx, my)
    acc = compute_grad(p) * 0.001
    tl.paintArrow(image, p, acc)
    paint_tree()

@ti.kernel
def advance():
    for i in range(N):
        acc = compute_grad(particles.pos[i]) * 0.01
        particles.vel[i] += acc * dt
    for i in range(N):
        particles.pos[i] += particles.vel[i] * dt
        particles.vel[i] = tl.boundReflect(particles.pos[i], particles.vel[i])


@ti.kernel
def compute_energy():
    kine_eng = 0.0
    pote_eng = 0.0
    for i in range(N):
        kine_eng += tl.sqrLength(particles.vel[i])
    for i in range(N):
        for j in range(i):
            pote_eng += tl.invLength(particles.pos[i] - particles.pos[j])
    print(kine_eng * 0.5 - pote_eng * 0.01)


## GUI Loop
init()
with ti.GUI('Tree-code Gravity', RES) as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        image.fill(0)
        for i in range(50):
            nodes.mpos.fill(0)
            nodes.mass.fill(0)
            tree.deactivate_all()
            build_tree()
            advance()
        compute_energy()
        #render(*gui.get_cursor_pos())
        gui.set_image(image)
        gui.circles(particles.pos.to_numpy(), radius=2)
        gui.show()
