import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec, vec2


class RawGrader(ts.DataOriented):
    @ti.func
    def compute_grad(self, pos):
        acc = vec2(0.0)
        for i in range(self.N):
            dir = self.pos[i] - pos
            dir = ts.normalizePow(dir, -2, 1e-3)
            acc += dir
        return acc


class FMMGrader(ts.DataOriented):
    def fmm_init(self, L=4, mps=4):
        self.level = L
        self.n_grid = 2 ** L
        self.grid_size = 1 / self.n_grid
        self.max_per_cell = mps
        # Level 0 is the minimal cells.
        # Level L-1 is four maxmimum cells.
        # Level L is the root cell.
        self.g_id = ti.var(ti.i32)
        self.tree = ti.root.bitmasked(ti.ij, self.n_grid)
        self.tree_ps = self.tree.dynamic(ti.k, self.max_per_cell)
        self.tree_ps.place(self.g_id)
        self.img = ti.var(ti.f32, (512, 512))

    @staticmethod
    @ti.func
    def which(above):
        ret = 0
        below = not above
        if below.x and below.y:
            ret = 0
        elif above.x and below.y:
            ret = 1
        elif below.x and above.y:
            ret = 2
        elif above.x and above.y:
            ret = 3
        return ret

    @ti.func
    def build_tree(self):
        for I in ti.grouped(self.tree):
            ti.deactivate(self.tree_ps, I)
        for i in range(self.N):
            P = int(self.pos[i] * self.n_grid)
            ti.append(self.tree_ps, P, i)

    @ti.func
    def paint_tree(self):
        for I in ti.grouped(self.img):
            self.img[I] = 0
        for I in ti.grouped(self.tree):
            gs = int(self.grid_size * 512)
            P = int(I * gs)
            for off in ti.grouped(ti.ndrange(gs, gs)):
                self.img[P.xy + off] = ti.length(self.tree_ps, I) / self.max_per_cell

    @ti.func
    def cell_id(self, level, offset):
        2 ** level

    @ti.func
    def compute_grad(self, pos):
        acc = vec2(0.0)
        for i in range(self.N):
            dir = self.pos[i] - pos
            dir = ts.normalizePow(dir, -2, 1e-3)
            acc += dir
        return acc


class MySolver(ts.Animation, FMMGrader):
    def on_init(self, N=3):
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
        self.pos[0] = vec(0.25, 0.46)
        self.pos[1] = vec(0.25, 0.54)
        self.vel[0] = vec(-1.5, +0.0)
        self.vel[1] = vec(+1.5, +0.0)
        self.pos[2] = vec(0.75, 0.5)

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
