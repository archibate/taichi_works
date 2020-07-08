import taichi as ti
import taichi_glsl as tl
ti.init()

kMaxParticles = 128
kResolution = 512
kKernelSize = 16 / 512

particle_pos = ti.Vector(2, ti.f32, kMaxParticles)
particle_vel = ti.Vector(2, ti.f32, kMaxParticles)
n_particles = ti.var(ti.i32, ())

image = ti.var(ti.f32, (kResolution, kResolution))


@ti.func
def smooth(distance):
    ret = 0.0
    r2 = distance.norm_sqr()
    if r2 < kKernelSize**2:
        ret = ti.exp(-r2 / kKernelSize**2) - ti.exp(-1)
    return ret


@ti.func
def alloc_particle():
    ret = ti.atomic_add(n_particles[None], 1)
    assert ret < kMaxParticles
    return ret


@ti.kernel
def add_particle_at(mx: ti.f32, my: ti.f32):
    id = alloc_particle()
    particle_pos[id] = tl.vec(mx, my)


@ti.kernel
def render_image():
    for pix in ti.grouped(image):
        pos = pix / kResolution
        field = 0.0
        for i in range(n_particles[None]):
            field += smooth(pos - particle_pos[i]) * 1.0
        image[pix] = field


gui = ti.GUI('WCSPH', kResolution)
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.RMB:
            add_particle_at(*gui.get_cursor_pos())

    if gui.is_pressed(gui.LMB):
        add_particle_at(*gui.get_cursor_pos())

    image.fill(0)
    render_image()
    gui.set_image(image)
    gui.show()
