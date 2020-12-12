from numbers import Number
import matplotlib.pyplot as plt

class StateSpace:

    x = []
    A = []
    B = []
    C = []
    D = []

    def __init__(self, x0, A, B=None, C=None, D=None):
        # rxc
        # x0 is the initial state
        # A is the state Matrix, nxn where n is length of x
        # B is the forcing Matrix, nxp
        # C is the output matrix, qxn
        # D is the feedthrough matrix qxp

        # validate everything

        n = len(x0)
        if B is None:
            B = Matrix([[0] for _ in range(n)])
        p = B.c
        if C is None:
            C = Matrix([n*[0]])
        q = C.r
        if D is None:
            D = Matrix([p*[0] for _ in range(q)])

        self.x = x0
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def step(self, u=None):
        if u is None:
            u = Vector([0])
        self.x = self.A@self.x + self.B@u
        y = self.C@self.x + self.D@u
        return y


class Vector:
    v = []

    def __init__(self, v, c=None):

        if isinstance(v, list):
            for n in v:
                if not isinstance(n, Number):
                    raise TypeError
            self.v = v
        elif isinstance(v, Matrix) and c is not None:
            self.v = [x[c] for x in v]
        else:
            raise TypeError

    def __len__(self):
        return len(self.v)
    
    def __add__(self, o):
        if len(self.v) != len(o.v):
            raise DimensionError
        return Vector([v1 + v2 for (v1, v2) in zip(self.v, o.v)])

    def __sub__(self, o):
        if len(self.v) != len(o.v):
            raise DimensionError
        return Vector([v1 - v2 for (v1, v2) in zip(self.v, o.v)])
    
    def __mul__(self, o):
        # dot product
        if isinstance(o, Vector):
            if len(self.v) != len(o.v):
                raise DimensionError
            sum = 0
            for (v1, v2) in zip(self.v, o.v):
                sum += v1*v2
            return sum
        elif isinstance(o, Number):
            return Vector([x*o for x in self])
        else:
            raise NotImplementedError
    
    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        if isinstance(o, Number):
            return self.__mul__(1/o)
        else:
            return NotImplementedError

    def __getitem__(self, index):
        return self.v[index]
    
    def __iter__(self):
        return iter(self.v)

    def __next__(self):
        return next(self.v)
    
    def __str__(self):
        s = '[\n'
        for x in self.v:
            s += str(x)
            s += '\n'
        s += ']'
        return s
    
    def __repr__(self):
        return "Vector: " + str(self)

    def __abs__(self):
        out = self*self
        return out**(1/2)
    
    def copy(self):
        return Vector(self.v.copy())
    
    def unit(self):
        return self/abs(self)


class Matrix:
    m = []
    r = 0
    c = 0

    def __init__(self, m):
        self.r = len(m)
        self.c = len(m[0])
        self.m = m
        self._validate_matrix()

    def __add__(self, o):
        if self.c != o.c or self.r != o.r:
            raise DimensionError

    def __matmul__(self, o):
        if isinstance(o, Vector):
            if len(o) != self.c:
                raise DimensionError
            return Vector([Vector(x)*o for x in self.m])
        elif isinstance(o, Matrix):
            if self.c != o.r:
                raise DimensionError
            matrixOut = []
            for k in self:
                matrixOut.append([Vector(k)*Vector(o, c) for c in range(o.c)])
            return Matrix(matrixOut)
        else:
            raise NotImplementedError
        
    def __mul__(self, o):
        if isinstance(o, Number):
            q = []
            for row in self:
                q.append([x*o for x in row])
            return Matrix(q)
        else:
            return NotImplementedError

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        if isinstance(o, Number):
            if o == 0:
                return float('inf')
            else:
                return self.__mul__(1/o)
        else:
            return NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, tuple):
            (r, c) = index
            return self.m[r][c]
        else:
            return self.m[index]
    
    def __iter__(self):
        return iter(self.m)

    def __next__(self):
        return next(self.m)

    def __str__(self):
        s = '[\n'
        for x in self.m:
            s += str(x)
            s += '\n'
        s += ']'
        return s
    
    def __repr__(self):
        return "Matrix: " + str(self)

    def _validate_matrix(self):
        # check that a matrix (a list of lists) has the correct number of rows and columns
        # we will at lease assume that the list is populated by lists or numbers
        if self.m is None and self.r != 0:
            raise DimensionError
        rows = len(self.m)
        if rows != self.r:
            print(f"Matrix: {self.m} doesn't have {self.r} rows")
            raise DimensionError

        matrix = True if isinstance(self.m[0], list) else False

        for m in self.m:
            if matrix and (len(m) != self.c):
                print(f"Matrix {self.m} doesn't have {self.c} columns")
                raise DimensionError


class DimensionError(Exception):
    pass





# Matrix defined as: [[a, b],
#                     [c, d]]
# (a list of rows)
# a list such as [a,
#                 b]
# can be assumed to be a column vector
# [[a],
#  [b]]
# is a valid column vector
# [[a, b]] is a valid row vector
# [] is 0xundefined, [[]] is 1x0
# [[],
#  []] is 2x0
# you can have 0x
# Assuming real numbers for now
# rows = len(M)
# columns = len(M[0])
# so you can have 


def twoBodyProblem():
    # one dimension
    fs = 1
    T = 1/fs

    m1 = 1
    m2 = 1
    G = 1

    r1 = Vector([1,
                 0,
                 0])

    v1 = Vector([0,
                 1,
                 0])

    r2 = Vector([-1,
                 0,
                 0])
    
    v2 = Vector([0,
                 1,
                 0])

    A = Matrix([[1, T, 0, 0, 0, 0],     # x
                [0, 1, 0, 0, 0, 0],     # vx
                [0, 0, 1, T, 0, 0],     # y  
                [0, 0, 0, 1, 0, 0],     # vy
                [0, 0, 0, 0, 1, T],     # z
                [0, 0, 0, 0, 0, 1]])    # vz

    B = Matrix([[0, 0, 0],
                [T, 0, 0],
                [0, 0, 0],
                [0, T, 0],
                [0, 0, 0],
                [0, 0, T]])

    C = Matrix([[1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0]])

    x1 = Vector([r1[0],     # x
                 v1[0],     # v_x
                 r1[1],     # y
                 v1[1],     # v_y
                 r1[2],     # z
                 v1[2]])    # z_y

    x2 = Vector([r2[0],     # x
                 v2[0],     # v_x
                 r2[1],     # y
                 v2[1],     # v_y 
                 r2[2],     # z
                 v2[2]])   # v_z
    
    s1 = StateSpace(x1, A, B, C)
    s2 = StateSpace(x2, A, B, C)

    o1 = [r1]
    o2 = [r2]
    for _ in range(200000):
        r12 = r1 - r2
        r21 = r2 - r1

        F12 = r12.unit()*-G*m2/(r12*r12)
        F21 = r21.unit()*-G*m1/(r21*r21)

        r1 = s1.step(F12)
        r2 = s2.step(F21)

        o1.append(r1)
        o2.append(r2)

    r = [{}, {}]
    
    r[0]['x'] = [q[0] for q in o1]
    r[0]['y'] = [q[1] for q in o1]
    r[0]['z'] = [q[2] for q in o1]

    r[1]['x'] = [q[0] for q in o2]
    r[1]['y'] = [q[1] for q in o2]
    r[1]['z'] = [q[2] for q in o2]

    return r

def main():
    r = twoBodyProblem()

    fig = plt.figure()
    plt.plot(r[0]['x'], r[0]['y'])
    plt.plot(r[1]['x'], r[1]['y'])
    plt.plot()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[0]['x'], r[0]['y'], r[0]['z'])
    ax.plot(r[1]['x'], r[1]['y'], r[1]['z'])
    plt.show()





if __name__ == "__main__":
    main()