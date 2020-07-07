import taichi as ti
import math


class vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __pos__(a):
        return vec(a.x, a.y)

    def __neg__(a):
        return vec(-a.x, -a.y)

    def __add__(a, b):
        return vec(a.x + b.x, a.y + b.y)

    __radd__ = __add__

    def __mul__(a, b):
        return vec(a.x * b, a.y * b)

    __rmul__ = __mul__

    def __truediv__(a, b):
        return a * (1 / b)

    def __repr__(self):
        return f'vec({self.x}, {self.y})'

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            raise IndexError

    def __len__(self):
        return 2


def pol(p, i):
    sqrt = math.sqrt
    x, y = p
    if i == (0, 0): ret = x/sqrt(x**2 + y**2)
    elif i == (1, 0): ret = -x**2/(x**2 + y**2)**(3/2) + 1/sqrt(x**2 + y**2)
    elif i == (1, 1): ret = -x*y/(x**2 + y**2)**(3/2)
    elif i == (2, 0): ret = 3*x**3/(x**2 + y**2)**(5/2) - 3*x/(x**2 + y**2)**(3/2)
    elif i == (2, 1): ret = 3*x**2*y/(x**2 + y**2)**(5/2) - y/(x**2 + y**2)**(3/2)
    elif i == (2, 3): ret = 3*x*y**2/(x**2 + y**2)**(5/2) - x/(x**2 + y**2)**(3/2)
    elif i == (3, 0): ret = -15*x**4/(x**2 + y**2)**(7/2) + 18*x**2/(x**2 + y**2)**(5/2) - 3/(x**2 + y**2)**(3/2)
    elif i == (3, 1): ret = -15*x**3*y/(x**2 + y**2)**(7/2) + 9*x*y/(x**2 + y**2)**(5/2)
    elif i == (3, 2): ret = -15*x**2*y**2/(x**2 + y**2)**(7/2) + 3*x**2/(x**2 + y**2)**(5/2) + 3*y**2/(x**2 + y**2)**(5/2) - 1/(x**2 + y**2)**(3/2)
    elif i == (3, 3): ret = -15*x*y**3/(x**2 + y**2)**(7/2) + 9*x*y/(x**2 + y**2)**(5/2)
    else: assert False, i
    return ret



N = 4

class multipole:
    def __init__(self, center=None, *q):
        self.center = center
        self.q = q

    def __repr__(self):
        return f'multipole({repr(self.center)}, {", ".join(map(repr, self.q))})'

    def translate(a, k, cent):
        cent = a.center - cent
        ret = 0
        for i in range(N - k - 1):
            ret += a.q[k - i] * cent ** k
        return ret

    def __add__(a, b):
        c = multipole()
        c.q = [0 for _ in range(N)]

        c.q[0] = a.q[0] + b.q[0]
        c.center = (a.q[0] * a.center + b.q[0] * b.center) / c.q[0]

        for k in range(1, len(c.q)):
            c.q[k] = a.translate(k, c.center) + b.translate(k, c.center)

        return c



"""
G = q g(r - r0)
G = q g(r) - q r0 @ g'(r) - q r0 @@ g''(r) - q r0 @@@ g'''(r)
"""

a = multipole(+1, 1, 0, 0, 0)
b = multipole(-1, 1, 0, 0, 0)
print(a + b)
