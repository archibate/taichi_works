import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec, vec2, vec3


@ti.func
def npow(x):
    d = ts.normalizePow(x, -2, 1e-3)
    if d.norm() > 1000:
        d = d * 0
    return d


class RawGrader(ts.DataOriented):
    def grader_init(self, *args, **kwargs):
        pass

    def build_tree(self):
        pass

    def paint_tree(self):
        pass

    @ti.func
    def compute_grad(self, pos):
        acc = vec2(0.0)
        for i in range(self.N):
            acc += npow(self.pos[i] - pos)
        return acc


class FMMNode(ts.TaichiClass):
    @property
    def mass(self):
        return self.entries[0]

    @property
    def ind(self):
        return self.entries[1]

    @property
    def CoM(self):
        return self.entries[2]

    @classmethod
    def make(cls):
        return cls(ti.var(ti.f32), ti.var(ti.i32), ti.Vector(2, ti.f32))


class FMMGrader(ts.DataOriented):
    def grader_init(self, L=2):
        self.n_level = L
        self.n_grid = 2 ** L
        self.grid_size = 1 / self.n_grid
        # Level 0 is the minimal nodes / grids.
        # Level L-1 is four maxmimum nodes.
        # Level L is the root node.
        self.levels = []
        self.nodes = []
        prev_level = ti.root
        for l in range(L):
            node = FMMNode.make()
            level = prev_level.bitmasked(ti.ij, 2)
            level.place(node)
            self.levels.append(level)
            self.nodes.append(node)
            prev_level = level
        self.img = ti.var(ti.f32, (512, 512))

    @ti.func
    def build_tree(self):
        for I in ti.grouped(self.tree):
            self.g_count[I] = 0
            ti.deactivate(self.tree, I)
        for i in range(self.N):
            P = int(self.pos[i] * self.n_grid)
            print('at', P, i)
            self.g_id[vec(P, ti.atomic_add(self.g_count[P], 1))] = i
        for I in ti.grouped(self.tree):
            self.g_mass[I] = self.cell_mass(I)
            self.g_CoM[I] = self.cell_CoM(I)

    @ti.func
    def paint_tree(self):
        for I in ti.grouped(self.img):
            self.img[I] = 0
        for I in ti.grouped(self.tree):
            gs = int(self.grid_size * 512)
            P = int(I * gs)
            for off in ti.grouped(ti.ndrange(gs, gs)):
                clr = self.g_count[I] / self.max_per_cell
                clr *= ti.exp(-100 * ts.sqrLength((P + off) / 512 - self.g_CoM[I]))
                self.img[P + off] = clr

    @ti.func
    def compute_grad(self, pos):
        acc = vec2(0.0)
        # account this cell:
        I = int(pos * self.n_grid)
        for c in range(self.g_count[I]):
            id = self.g_id[vec(I, c)]
            d = self.pos[id] - pos
            acc += npow(d)
        for J in ti.grouped(ti.ndrange(self.n_grid, self.n_grid)):
            if all(J != I):
                d = self.g_CoM[J] - pos
                acc += self.g_mass[J] * npow(d)
        return acc


class MySolver(ts.Animation, FFMGrader):
    def on_init(self, N=3):
        self.N = N
        self.pos = ti.Vector(2, ti.f32, N)
        self.vel = ti.Vector(2, ti.f32, N)
        self.circles = self.pos
        self.circle_radius = 2
        self.dt = 0.002
        self.grader_init()

    @ti.kernel
    def on_start(self):
        #for i in self.pos:
        #    self.pos[i] = ts.randND(2)
        if ti.static(0):
            self.pos[0] = vec(0.25, 0.46)
            self.pos[1] = vec(0.25, 0.54)
            self.vel[0] = vec(-1.5, +0.0)
            self.vel[1] = vec(+1.5, +0.0)
            self.pos[2] = vec(0.75, 0.5)
        if ti.static(1):
            self.pos[0] = vec(0.4, 0.34)
            self.pos[1] = vec(0.4, 0.38)
            self.vel[0] = vec(-1.0, +0.0)
            self.vel[1] = vec(+1.0, +0.0)
            self.pos[2] = vec(0.71, 0.6)
            self.vel[2] = vec(-0.1, -0.3)
        if ti.static(0):
            self.pos[0] = vec(0.4, 0.34)
            self.vel[0] = vec(+0.0, +0.0)
            self.pos[1] = vec(0.71, 0.55)
            self.vel[1] = vec(-0.1, -0.3)

    @ti.kernel
    def on_advance(self):
        self.build_tree()
        for i in self.pos:
            acc = self.compute_grad(self.pos[i])
            self.vel[i] += acc * self.dt
        for i in self.pos:
            self.pos[i] += self.vel[i] * self.dt

    @ti.kernel
    def on_render(self):
        self.paint_tree()



ti.init()
MySolver().start()
