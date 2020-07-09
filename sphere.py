import taichi as ti
import taichi_glsl as tl
import random, math
ti.init()#kernel_profiler=True)

dt = 0.01
kMaxParticles = 1024
kResolution = 512
kKernelSize = 16 / 512
kKernelFactor = 0.5 / kKernelSize**2
kGravity = tl.vec(0.0, -0.0)

kUseImage = False
kBackgroundColor = 0x112f41
kParticleColor = 0x068587
kBoundaryColor = 0xebaca2
kParticleDisplaySize = 0.2 * kKernelSize * kResolution

particle_pos = ti.Vector(2, ti.f32, kMaxParticles)
particle_vel = ti.Vector(2, ti.f32, kMaxParticles)
property_vel = ti.Vector(2, ti.f32, kMaxParticles)
property_density = ti.var(ti.f32, kMaxParticles)
property_force = ti.Vector(2, ti.f32, kMaxParticles)
n_particles = ti.var(ti.i32, ())

if kUseImage:
    image = ti.Vector(3, ti.f32, (kResolution, kResolution))


@ti.func
def smooth(distance):
    ret = 0.0
    r2 = distance.norm_sqr()
    if r2 < kKernelSize**2:
        ret = ti.exp(-r2 * kKernelFactor)
    return ret


@ti.func
def grad_smooth(distance):
    ret = tl.vec2(0.0)
    r2 = distance.norm_sqr()
    if r2 < kKernelSize**2:
        ret = (-2 * kKernelFactor) * distance * ti.exp(-r2 * kKernelFactor)
    return ret


@ti.func
def alloc_particle():
    ret = ti.atomic_add(n_particles[None], 1)
    assert ret < kMaxParticles
    return ret


@ti.kernel
def add_particle_at(mx: ti.f32, my: ti.f32, vx: ti.f32, vy: ti.f32):
    id = alloc_particle()
    particle_pos[id] = tl.vec(mx, my)
    particle_vel[id] = tl.vec(vx, vy)


@ti.func
def preupdate(rho, rho_0=1000, gamma=7.0, c_0=20.0):
    b = rho_0 * c_0**2 / gamma
    return b * ((rho / rho_0) ** gamma - 1.0)


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
    for i in range(n_particles[None]):
        my_pos = particle_pos[i]
        property_force[i] = tl.vec2(0.0)
        for j in range(n_particles[None]):
            dw = grad_smooth(my_pos - particle_pos[j])
            ds = particle_pos[j] - particle_pos[i]
            dv = particle_vel[j] - particle_vel[i]
            force = dw * property_density[j] * dv.dot(ds)
            property_force[i] += force


@ti.kernel
def substep():
    update_property()
    for i in range(n_particles[None]):
        gravity = (0.5 - particle_pos[i]) * 2.0
        particle_vel[i] += gravity * dt
        particle_vel[i] += property_force[i] * dt
        particle_vel[i] = tl.boundReflect(particle_pos[i], particle_vel[i],
                                          kKernelSize, 1 - kKernelSize, 0)
        particle_pos[i] += particle_vel[i] * dt
        particle_pressure[i] = preupdate(particle_density)


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
        elif e.key == 'r':
            a = random.random() * math.tau
            add_particle_at(math.cos(a) * 0.4 + 0.5, math.sin(a) * 0.4 + 0.5, 0, 0)

    substep()
    if kUseImage:
        update_image()
        gui.set_image(image)
    else:
        gui.circles(particle_pos.to_numpy()[:n_particles[None]],
                    radius=kParticleDisplaySize, color=kParticleColor)
    gui.show()
