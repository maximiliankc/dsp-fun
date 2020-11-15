from numbers import Number


class StateSpace:

    x = []
    A = []
    B = []
    C = []
    D = []
    n = 0
    p = 0
    q = 0

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
            B = n*[[0]]
        p = len(B[0])
        if C is None:
            C = [n*[0]]
        q = len(C)
        if D is None:
            D = q*[p*[0]]

        validate_matrix(A, n, n)
        validate_matrix(B, n, p)
        validate_matrix(C, q, n)
        validate_matrix(D, q, p)

        print(f'A: {A}')
        print(f'B: {B}')
        print(f'C: {C}')
        print(f'D: {D}')

        self.x = x0
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n = n
        self.p = p
        self.q = q


    def step(self, u=[0]):
        if len(u) != self.p:
            print(f"vector {u} doesn't have length {self.p}")
            raise DimensionError
        self.x = vector_add(matrix_multiply(self.x, self.A), matrix_multiply(u, self.B))
        y = vector_add(matrix_multiply(self.x, self.C), matrix_multiply(u, self.D))
        return (self.x.copy(), y)


class Vector:
    v = []

    def __init__(self, v, c = None):

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
        if len(self.v) != len(o.v):
           raise DimensionError
        sum = 0
        for (v1, v2) in zip(self.v, o.v):
            sum += v1*v2
        return sum

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


class Matrix:
    m = []
    r = 0
    c = 0

    def __init__(self, m):
        self.r = len(m)
        self.c = len(m[0])
        validate_matrix(m, self.r, self.c)
        self.m = m

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

def validate_matrix(M, r, c):
    # check that a matrix (a list of lists) has the correct number of rows and columns
    # we will at lease assume that the list is populated by lists or numbers
    if M is None and r != 0:
        raise DimensionError
    rows = len(M)
    if rows != r:
        print(f"Matrix: {M} doesn't have {r} rows")
        raise DimensionError

    matrix = True if isinstance(M[0], list) else False

    for m in M:
        if matrix and (len(m) != c):
            print(f"Matrix {M} doesn't have {c} columns")
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


def main():
    x = Vector([1, 0])
    y = Vector([3, 1])
    A = Matrix([[1, 2],
                [3, 4]])
    B = Matrix([[5, 6],
                [7, 8]])
    C = Matrix([[9, 10],
                [11, 12]])

    I = Matrix([[1,0],
                [0,1]])

    X = Vector([1,1])
    Y = Matrix([[1,2],
                [3,4],
                [5,6]])

    print(f'I squared:\n{I@I}')
    print(f'AB:\n{A@B}')
    print(f'YA\n{Y@A}')



    # print(x + y)
    # print(x*y)

    # print(f"x: {x}")
    # print(f"A: {A}")
    # print(f"B: {B}")
    # print(f"C: {C}")

    # print(f"A@x {A@x}")
    # print(f"B@x {B@x}")
    # print(f"C@x {C@x}")

    #s = StateSpace(x0, A, B, C)
    #s.step([1, 2])

if __name__ == "__main__":
    main()