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


class FMMGrader(ts.DataOriented):
    def fmm_init(self, L=2, mps=4):
        self.level = L
        self.n_grid = 2 ** L
        self.grid_size = 1 / self.n_grid
        self.max_per_cell = mps
        # Level 0 is the minimal nodes / cells.
        # Level L-1 is four maxmimum nodes.
        # Level L is the root node.
        self.g_id = ti.var(ti.i32)
        self.g_mass = ti.var(ti.f32)
        self.g_CoM = ti.Vector(2, ti.f32)
        self.tree = ti.root.bitmasked(ti.ij, self.n_grid)
        self.tree.place(self.g_mass, self.g_CoM)
        self.tree_ps = self.tree.dynamic(ti.k, self.max_per_cell)
        self.tree_ps.place(self.g_id)
        self.img = ti.var(ti.f32, (512, 512))

    @ti.func
    def build_tree(self):
        for I in ti.grouped(self.tree):
            ti.deactivate(self.tree_ps, I)
        for i in range(self.N):
            P = int(self.pos[i] * self.n_grid)
            print('at', P, i)
            ti.append(self.tree_ps, P, i)
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
                clr = ti.length(self.tree_ps, I) / self.max_per_cell
                clr *= ti.exp(-100 * ts.sqrLength((P + off) / 512 - self.g_CoM[I]))
                self.img[P + off] = clr

    @ti.func
    def cell_mass(self, I):
        mass = 0.0
        len = ti.length(self.tree_ps, I)
        for c in range(len):
            mass += 1.0
        return mass

    @ti.func
    def cell_CoM(self, I):
        center = vec2(0.0)
        len = ti.length(self.tree_ps, I)
        for c in range(len):
            id = self.g_id[vec(I, c)]
            center += self.pos[id]
        center *= 1 / len
        return center

    @ti.func
    def compute_grad(self, pos, ki):
        acc = vec2(0.0)
        # account this cell:
        I = int(pos * self.n_grid)
        len = ti.length(self.tree_ps, I)
        for c in range(len):
            id = self.g_id[vec(I, c)]
            if ki != id:
                print('cg', ki, id)
                acc += npow(self.pos[id] - pos)
        return acc


class MySolver(ts.Animation, FMMGrader):
    def on_init(self, N=2):
        self.N = N
        self.pos = ti.Vector(2, ti.f32, N)
        self.vel = ti.Vector(2, ti.f32, N)
        self.circles = self.pos
        self.circle_radius = 2
        self.dt = 0.002
        self.fmm_init()

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
        if ti.static(0):
            self.pos[0] = vec(0.4, 0.34)
            self.pos[1] = vec(0.4, 0.38)
            self.vel[0] = vec(-1.0, +0.0)
            self.vel[1] = vec(+1.0, +0.0)
            self.pos[2] = vec(0.71, 0.6)
            self.vel[2] = vec(-0.1, -0.3)
        if ti.static(1):
            self.pos[0] = vec(0.4, 0.34)
            self.vel[0] = vec(+0.0, +0.0)
            self.pos[1] = vec(0.71, 0.6)
            self.vel[1] = vec(-0.1, -0.3)

    @ti.kernel
    def on_advance(self):
        self.build_tree()
        for i in self.pos:
            acc = self.compute_grad(self.pos[i], i)
            self.vel[i] += acc * self.dt
        for i in self.pos:
            self.pos[i] += self.vel[i] * self.dt

    @ti.kernel
    def on_render(self):
        self.paint_tree()



ti.init()
MySolver().start()
