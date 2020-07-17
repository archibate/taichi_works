import taichi as ti
import taichi_glsl as tl
ti.init()

NF = 6
NV = NF + 2
E, nu = 5e3, 0.2
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)

vertices = ti.Vector.var(2, ti.f32, NV)
faces = ti.Vector.var(3, ti.i32, NF)
B = ti.Matrix.var(2, 2, ti.f32, NF)
F = ti.Matrix.var(2, 2, ti.f32, NF)
phi = ti.var(ti.f32, NF)


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
        a = vertices[faces[i].x]
        b = vertices[faces[i].y]
        c = vertices[faces[i].z]
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]


@ti.kernel
def update_P():
    for i in range(NF):
        F_i = F[i]
        J_i = F_i.determinant()
        log_J_i = ti.log(J_i)
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
        phi_i -= mu * log_J_i
        phi_i += lam / 2 * log_J_i ** 2
        phi[i] = phi_i


def paint_P():
    for i in range(NF):
        a = vertices[faces[i].x]
        b = vertices[faces[i].y]
        c = vertices[faces[i].z]
        phi_i = phi[i]
        color = phi_i * tl.D.xyy + (1 - phi_i) * tl.D.yxy
        try:
            gui.triangle(a, b, c, color=ti.rgb_to_hex(color))
        except ValueError:
            gui.triangle(a, b, c)


@ti.kernel
def init():
    for i in range(NF):
        faces[i] = [i, i + 1, i + 2]
    for i in range(NV):
        vertices[i] = tl.vec(i / NV * 0.8 + 0.1,
                0.5 + ti.pow(-1, i) * (ti.random() * 0.2 + 0.9) * 0.15)


init()
update_B()
gui = ti.GUI('FEM')
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False

    update_F()
    update_P()
    paint_P()

    gui.circles(vertices.to_numpy(), radius=4)
    gui.show()
