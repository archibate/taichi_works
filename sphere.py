import taichi as ti
import taichi_glsl as tl
ti.init(kernel_profiler=True)

dt = 0.001
kMaxParticles = 128
kResolution = 512
kKernelSize = 16 / 512
kGravity = tl.vec(0, -0.1)

kUseImage = False
kBackgroundColor = 0x112f41
kParticleColor = 0x068587
kBoundaryColor = 0xebaca2

particle_pos = ti.Vector(2, ti.f32, kMaxParticles)
particle_vel = ti.Vector(2, ti.f32, kMaxParticles)
property_vel = ti.Vector(2, ti.f32, kMaxParticles)
property_density = ti.var(ti.f32, kMaxParticles)
n_particles = ti.var(ti.i32, ())

if kUseImage:
    image = ti.Vector(3, ti.f32, (kResolution, kResolution))


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


@ti.func
def update_property():
    for i in range(n_particles[None]):
        my_pos = particle_pos[i]
        property_vel[i] = particle_vel[i]
        property_density[i] = 1.0
        for j in range(n_particles[None]):
            w = smooth(my_pos - particle_pos[j])
            property_vel[i] += w * particle_vel[j]
            property_density[i] += w
        property_vel[i] /= property_density[i]


@ti.kernel
def substep():
    update_property()
    property_vel[0] += kGravity
    for i in range(n_particles[None]):
        particle_vel[i] = property_vel[i]
        particle_vel[i] = tl.boundReflect(particle_pos[i], particle_vel[i],
                                          kKernelSize, 1 - kKernelSize, 0)
        particle_pos[i] += particle_vel[i] * dt


@ti.kernel
def update_image():
    for i in ti.grouped(image):
        image[i] = tl.vec3(0)
    for i in range(n_particles[None]):
        pos = particle_pos[i]
        A = ti.floor(max(0, pos - kKernelSize)) * kResolution
        B = ti.ceil(min(1, pos + kKernelSize + 1)) * kResolution
        for pix in ti.grouped(ti.ndrange((A.x, B.x), (A.y, B.y))):
            pix_pos = pix / kResolution
            w = smooth(pix_pos - particle_pos[i])
            image[pix].x += w


gui = ti.GUI('WCSPH', kResolution, background_color=kBackgroundColor)
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.RMB:
            add_particle_at(*gui.get_cursor_pos())

    if gui.is_pressed(gui.LMB):
        add_particle_at(*gui.get_cursor_pos())

    substep()
    if kUseImage:
        update_image()
        gui.set_image(image)
    else:
        gui.circles(particle_pos.to_numpy()[:n_particles[None]],
                    radius=kKernelSize * kResolution, color=kParticleColor)
    gui.show()

ti.kernel_profiler_print()
