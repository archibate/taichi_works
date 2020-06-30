## Initialization
import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec, vec2, vec3, math
ti.init()

## Define Variables
N = 12
pos = ti.Vector(2, ti.f32, N)
img = ti.var(ti.f32, (512, 512))



## Helper Functions
@ti.func
def npow(x):
    d = ts.normalizePow(x, -2, 1e-3)
    if d.norm() > 1000:
        d = d * 0
    return d


@ti.func
def sdLine(u, v, p):
    pu = p - u
    vp = v - p
    vu = v - u
    puXvu = pu.cross(vu)
    puOvu = pu.dot(vu)
    vpOvu = vp.dot(vu)
    ret = 0.0
    if puOvu < 0:
        ret = ts.length(pu)
    elif vpOvu < 0:
        ret = ts.length(vp)
    else:
        ret = puXvu * ts.invLength(vu)
    return ret

@ti.func
def paintArrow(img: ti.template(), orig, dir):
    res = vec(*img.shape)
    I = orig * res
    D = dir * res
    J = I + D
    W = 3
    S = min(22, ts.length(D) * 0.5)
    DS = ts.normalize(D) * S
    SW = S + W
    D1 = ti.Matrix.rotation2d(+math.pi * 3 / 4) @ DS
    D2 = ti.Matrix.rotation2d(-math.pi * 3 / 4) @ DS
    bmin, bmax = int(ti.floor(min(I, J))), int(ti.ceil(max(I, J)))
    for P in ti.grouped(ti.ndrange((bmin.x - SW, bmax.x + SW), (bmin.y - SW, bmax.y + SW))):
        c0 = ts.smoothstep(abs(sdLine(I, J, P)), W, W / 2)
        c1 = ts.smoothstep(abs(sdLine(J, J + D1, P)), W, W / 2)
        c2 = ts.smoothstep(abs(sdLine(J, J + D2, P)), W, W / 2)
        img[P] = max(c0, c1, c2)


## Main Program
@ti.kernel
def init():
    for i in pos:
        pos[i] = ts.randND(2)


@ti.func
def compute_grad(pos):
    return (pos - 0.5).yx * vec(-1, 1)


@ti.kernel
def render(mx: ti.f32, my: ti.f32):
    pos = vec(mx, my)
    acc = compute_grad(pos)
    paintArrow(img, pos, acc)


## GUI Loop
with ti.GUI('FFM Gravity') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        img.fill(0)
        render(*gui.get_cursor_pos())
        gui.circles(pos.to_numpy(), radius=2)
        gui.set_image(img)
        gui.show()
