import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec, vec2


class FMMGrader(ts.DataOriented):
    def fmm_init(self, L=4):
        self.g_id = ti.var(ti.i32)
        self.tree = ti.root.bitmasked(ti.ij, 2 ** (L + 1))
        self.tree.place(self.g_id)

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
        for i in range(self.N):
            self.g_id[P] = i

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
        for i in self.pos:
            acc = self.compute_grad(self.pos[i])
            self.vel[i] += acc * self.dt
        for i in self.pos:
            self.pos[i] += self.vel[i] * self.dt



ti.init()
MySolver().start()
