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
            B = Matrix(n*[[0]])
        p = B.c
        if C is None:
            C = Matrix([n*[0]])
        q = C.r
        if D is None:
            D = Matrix(q*[p*[0]])

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
        return (self.x.copy(), y)


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
    
    def copy(self):
        return Vector(self.v.copy())


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
    pass


def main():
    x0 = Vector([0,0])
    fs = 1000
    A = Matrix([[1, 1/fs],
                [0, 1]])
    B = Matrix([[0],
                [1/fs]])
    g = Vector([-9.8])
    model = StateSpace(x0, A, B)
    y = []
    for _ in range(1):
        y.append(model.step(g))
    
    x = [v for (v,_) in y]

    plt.plot(x)
    plt.show()



if __name__ == "__main__":
    main()