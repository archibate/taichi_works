import taichi as ti
import taichi_glsl as tl

N = 8

x = ti.var(ti.f32, N)
m = ti.var(ti.f32, (N, N))
r = ti.var(ti.f32, (N, N))
ma = ti.var(ti.f32, N)
mc = ti.var(ti.f32, N)


@ti.kernel
def init():
    for i in range(N):
        m[i, i] = 2
        m[(i - 1) % N, i] = -1
        m[(i + 1) % N, i] = -1


def show(m):
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
def cyclic(i: ti.i32, s: ti.i32, o: ti.i32, n: ti.template()):
    a = i
    b, c = (i + tl.vec(1, 2)) % N
    k = -m[s * b + o, s * a + o] / m[s * a + o, s * a + o]
    for t in ti.static(range(n)):
        ma[t] = m[a, t] * k
    k = -m[s * b + o, s * c + o] / m[s * c + o, s * c + o]
    for t in ti.static(range(n)):
        mc[t] = m[s * c + o, t] * k

    for t in ti.static(range(n)):
        r[s * b + o, s * t + o] = m[s * b + o, s * t + o] + ma[t] + mc[t]


init()
for i in range(N):
    cyclic(i, 1, 0, N)
m.from_numpy(r.to_numpy())
show(m)
print('=========')
for i in range(N // 2):
    for j in range(2):
        cyclic(i, 2, j, N // 2)
m.from_numpy(r.to_numpy())
show(m)
