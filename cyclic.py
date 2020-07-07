import taichi as ti
import taichi_glsl as tl

N = 8

x = ti.var(ti.f32, N)
m = ti.var(ti.f32, (N, N))


@ti.kernel
def init():
    for i in range(N):
        m[i, i] = 2
        m[(i - 1) % N, i] = -1
        m[(i + 1) % N, i] = -1


def show():
    for i in range(N):
        for j in range(N):
            if m[i, j] != 0:
                print(f'{m[i, j]:-8.1f}', end='')
            else:
                print(f'      . ', end='')
        print('')


@ti.kernel
def cyclic():
    k = -m[3, 2] / m[2, 2]
    for i in ti.static(range(-1, 2)):
        m[2, 2 + i] *= k
    k = -m[3, 4] / m[4, 4]
    for i in ti.static(range(-1, 2)):
        m[4, 4 + i] *= k

    for i in ti.static(range(-2, 3)):
        m[3, 3 + i] += m[2, 3 + i] + m[4, 3 + i]


@ti.kernel
def cyclic():
    i = 2
    a = i
    b, c = (i + tl.vec(1, 2)) % N
    k = -m[b, a] / m[a, a]
    for t in ti.static(range(-2, 3)):
        m[a, a + t] *= k
    k = -m[b, c] / m[c, c]
    for t in ti.static(range(-2, 3)):
        m[c, c + t] *= k

    for t in ti.static(range(-2, 3)):
        m[b, b + t] += m[a, b + t] + m[c, b + t]


init()
cyclic()
show()
