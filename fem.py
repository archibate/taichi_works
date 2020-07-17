import taichi as ti
import taichi_glsl as tl
ti.init()

NF = 6
NV = NF + 2
E, nu = 5e3, 0.2
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
damping = 100.0
dt = 0.001

vertices = ti.Vector.var(2, ti.f32, NV)
velocity = ti.Vector.var(2, ti.f32, NV)
faces = ti.Vector.var(3, ti.i32, NF)
B = ti.Matrix.var(2, 2, ti.f32, NF)
F = ti.Matrix.var(2, 2, ti.f32, NF)
dF = ti.Vector.var(2, ti.f32, (NF, NV))
P = ti.Matrix.var(2, 2, ti.f32, NF)
phi = ti.var(ti.f32, NF)
V = ti.var(ti.f32, NF)


@ti.kernel
def update_B():
    for i in range(NF):
        a = vertices[faces[i].x]
        b = vertices[faces[i].y]
        c = vertices[faces[i].z]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()

@ti.kernel
def update_F():
    for i in range(NF):
        ia, ib, ic = faces[i]
        a = vertices[ia]
        b = vertices[ib]
        c = vertices[ic]
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]
        dF[i, ia] += B[i] @ ti.Vector([1, 0])
        dF[i, ib] += B[i] @ ti.Vector([0, 1])
        dF[i, ic] += B[i] @ ti.Vector([-1, -1])
        V[i] = abs((a - c).cross(b - c))


@ti.kernel
def update_P():
    for i in range(NF):
        F_i = F[i]
        J_i = F_i.determinant()
        log_J_i = ti.log(J_i)
        P_i = mu * (F_i - F_i.transpose())
        P[i] = P_i


@ti.kernel
def advance():
    for i in range(NV):
        f_i = tl.vec2(0.0)
        for e in range(NF):
            f_i += V[e] * P[e] @ dF[e, i]
        velocity[i] += -dt * f_i
        velocity[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        vertices[i] += dt * velocity[i]


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


def paint_phi():
    for i in range(NF):
        a = vertices[faces[i].x]
        b = vertices[faces[i].y]
        c = vertices[faces[i].z]
        k = phi[i] * 0.002
        color = k * tl.D.xyy + (1 - k) * tl.D.yxy
        try:
            gui.triangle(a, b, c, color=ti.rgb_to_hex(color))
        except ValueError:
            gui.triangle(a, b, c)


def pull(m0):
    closest = -1
    closest_dist = 1e5
    for i in range(NV):
        dist = (m0 - vertices[i].value).L
        if dist < closest_dist:
            closest, closest_dist = i, dist

    vertices[closest] = m0.entries


@ti.kernel
def init():
    for i in range(NF):
        faces[i] = [i, i + 1, i + 2]
    for i in range(NV):
        vertices[i] = tl.vec(i / NV * 0.8 + 0.1,
                0.5 + ti.pow(-1, i) * (ti.random() * 0.0 + 0.9) * 0.15)


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

    dF.fill(0)
    update_F()
    update_P()
    advance()
    update_phi()
    paint_phi()

    gui.circles(vertices.to_numpy(), radius=4)
    gui.show()
