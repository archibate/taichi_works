import taichi as ti
import taichi_glsl as tl
ti.init()

N = 4
NF = 2 * N ** 2
NV = (N + 1) ** 2
E, nu = 5e3, 0.2
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
gravity = tl.vec(0, -40)
damping = 20.0
dt = 5e-4

ball_pos = tl.vec(0.5, 0.0)
ball_radius = 0.35

pos = ti.Vector.var(2, ti.f32, NV, needs_grad=True)
vel = ti.Vector.var(2, ti.f32, NV)
faces = ti.Vector.var(3, ti.i32, NF)
B = ti.Matrix.var(2, 2, ti.f32, NF)
F = ti.Matrix.var(2, 2, ti.f32, NF, needs_grad=True)
V = ti.var(ti.f32, NF)
phi = ti.var(ti.f32, NF)
U = ti.var(ti.f32, (), needs_grad=True)


@ti.kernel
def update_B():
    for i in range(NF):
        a = pos[faces[i].x]
        b = pos[faces[i].y]
        c = pos[faces[i].z]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()


@ti.kernel
def update_F():
    for i in range(NF):
        ia, ib, ic = faces[i]
        a = pos[ia]
        b = pos[ib]
        c = pos[ic]
        V[i] = abs((a - c).cross(b - c))
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]


@ti.kernel
def update_phi():
    for i in range(NF):
        F_i = F[i]
        J_i = F_i.determinant()
        log_J_i = ti.log(J_i)
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
        phi_i -= mu * log_J_i
        phi_i += lam / 2 * log_J_i ** 2
        phi[i] = phi_i
        U[None] += V[i] * phi_i


@ti.kernel
def advance():
    for i in range(NV):
        f_i = -pos.grad[i]
        vel[i] += dt * (f_i + gravity)
        vel[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        vel[i] = tl.ballBoundReflect(pos[i], vel[i], ball_pos, ball_radius)
        vel[i] = tl.boundReflect(pos[i], vel[i], 0, 1, 0)
        pos[i] += dt * vel[i]


def paint_phi():
    for i in range(NF):
        a = pos[faces[i].x]
        b = pos[faces[i].y]
        c = pos[faces[i].z]
        k = phi[i] * 0.001
        color = k * tl.D.xyy + (1 - k) * tl.D.xxx * 0.5
        try:
            gui.triangle(a, b, c, color=ti.rgb_to_hex(color))
        except ValueError:
            gui.triangle(a, b, c)


def pull(m0):
    closest = -1
    closest_dist = 1e5
    for i in range(NV):
        dist = (m0 - pos[i].value).L
        if dist < closest_dist:
            closest, closest_dist = i, dist

    pos[closest] = m0.entries


@ti.kernel
def init():
    for i, j in ti.ndrange(N + 1, N + 1):
        pos[i * (N + 1) + j] = tl.vec(i, j) / N * 0.25 + 0.5 
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        faces[k + 0] = [i * (N + 1) + j, (i + 1) * (N + 1) + j, (i + 1) * (N + 1) + (j + 1)]
        faces[k + 1] = [(i + 1) * (N + 1) + (j + 1), i * (N + 1) + j, i * (N + 1) + (j + 1)]


def substep():
    with ti.Tape(loss=U):
        update_F()
        update_phi()
    advance()


init()
update_B()
gui = ti.GUI('FEM')
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.LMB:
            pull(tl.vec(*gui.get_cursor_pos()))
            m0 = None

    for i in range(10):
        substep()

    paint_phi()

    gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
    gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
    gui.show()
