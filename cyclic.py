import taichi as ti

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
            print(f'{m[i, j]:-8.2f}', end='')
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


init()
cyclic()
show()
