import taichi as ti
import taichi_glsl as tl
import random, math
ti.init()#kernel_profiler=True)

dt = 0.01
dx = 0.2
h0 = 0.1
gamma = 7
c_0 = 20.0
rho_0 = 1000.0
m0 = dx**2 * 100
kMaxParticles = 1024
kResolution = 512

kBackgroundColor = 0x112f41
kParticleColor = 0x068587
kBoundaryColor = 0xebaca2
kParticleSize = 5

pos = ti.Vector(2, ti.f32, kMaxParticles)
vel = ti.Vector(2, ti.f32, kMaxParticles)
pressure = ti.var(ti.f32, kMaxParticles)
density = ti.var(ti.f32, kMaxParticles)
d_vel = ti.Vector(2, ti.f32, kMaxParticles)
d_pressure = ti.var(ti.f32, kMaxParticles)
d_density = ti.var(ti.f32, kMaxParticles)
num = ti.var(ti.i32, ())


@ti.func
def alloc_particle():
    ret = ti.atomic_add(num[None], 1)
    assert ret < kMaxParticles
    return ret


@ti.func
def cubic(r):
    k = 10. / (7 * math.pi * h0 ** 2)
    q = r / h0
    assert q >= 0.0
    res = 0.0
    if q <= 1.0:
        res = k * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
    elif q < 2.0:
        res = k * 0.25 * (2 - q) ** 3
    return res

@ti.func
def grad_cubic(r):
    k = 10. / (7 * math.pi * h0 ** 2)
    q = r / h0
    assert q > 0.0
    res = 0.0
    if q < 1.0:
        res = (k / h0) * (-3 * q + 2.25 * q ** 2)
    elif q < 2.0:
        res = -0.75 * (k / h0) * (2 - q) ** 2
    return res


@ti.func
def grad_rho(i, j, r, r_mod):
    return m0 * grad_cubic(r_mod) * (vel[i] - vel[j]).dot(r / r_mod)


@ti.func
def tait(rho):
    b = rho_0 * c_0**2 / gamma
    return b * ((rho / rho_0) ** gamma - 1.0)


@ti.func
def pressure_force(i, j, r, r_mod):
    return -m0 * (pressure[i] / density[i]**2
                + pressure[i] / rho_0) * cubic(r_mod) * r / r_mod


@ti.func
def update_deltas():
    for i in range(num[None]):
        d_v = tl.vec2(0.0)
        d_rho = 0.0
        d_v = (0.5 - pos[i]) * 1.0
        for j in range(num[None]):
            r = pos[i] - pos[j]
            r_mod = r.norm()
            d_rho += grad_rho(i, j, r, r_mod)
            d_v += pressure_force(i, j, r, r_mod)

        d_vel[i] = d_v
        d_density[i] = d_rho


@ti.kernel
def substep():
    update_deltas()
    for i in range(num[None]):
        vel[i] += dt * d_vel[i]
        density[i] += dt * d_density[i]
        pressure[i] = tait(density[i])
        pos[i] += dt * vel[i]


@ti.kernel
def add_particle_at(mx: ti.f32, my: ti.f32, vx: ti.f32, vy: ti.f32):
    id = alloc_particle()
    pos[id] = tl.vec(mx, my)
    vel[id] = tl.vec(vx, vy)


last_mouse = tl.vec2(0.0)

gui = ti.GUI('WCSPH', kResolution, background_color=kBackgroundColor)
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.LMB:
            if e.type == gui.PRESS:
                last_mouse = tl.vec(*gui.get_cursor_pos())
            else:
                mouse = tl.vec(*gui.get_cursor_pos())
                diff = (mouse - last_mouse) * 2.0
                add_particle_at(mouse.x, mouse.y, diff.x, diff.y)
        elif e.type == gui.PRESS and e.key == 'r':
                a = random.random() * math.tau
                add_particle_at(math.cos(a) * 0.4 + 0.5, math.sin(a) * 0.4 + 0.5, 0, 0)

    substep()
    gui.circles(pos.to_numpy()[:num[None]],
                radius=kParticleSize, color=kParticleColor)
    gui.show()
