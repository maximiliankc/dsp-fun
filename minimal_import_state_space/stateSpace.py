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

    def __init__(self, v):
        if not isinstance(v, list):
            raise TypeError
        for n in v:
            if not isinstance(n, Number):
                raise TypeError
        self.v = v

    def __len__(self):
        return len(self.v)
    
    def __add__(self, o):
        if len(self.v) != len(o.v):
            raise DimensionError
        return [v1 + v2 for (v1, v2) in zip(self.v, o.v)]
    
    def __mul__(self, o):
        if len(self.v) != len(o.v):
           raise DimensionError
        sum = 0
        for (v1, v2) in zip(self.v, o.v):
            sum += v1*v2
        return sum

    def __getitem__(self, index):
        pass

class Matrix:
    m = []
    r = 0
    c = 0

    def __init__(self, m):
        pass

    def __add__(self, o):
        pass

    def __matmul__(self, o):




def dot(V1, V2):
    sum = 0
    for (v1, v2) in zip(V1, V2):
        sum += v1*v2
    return sum
        
def vector_add(v1, v2):
    return [x + y for (x, y) in zip(v1, v2)]

def matrix_multiply(M1, M2):
    # M2 can be a column vector, M1 cannot
    cM1 = len(M1[0])
    rM1 = len(M1)
    validate_matrix(M1, rM1, cM1)
    rM2 = len(M2)
    if cM1 != rM2:
        print(f"M1 has {cM1} columns, M2 has {rM2} rows")
        raise DimensionError

    matrix = True if isinstance(M2[0], list) else False

    if matrix:
        cM2 = len(M2[0])
        validate_matrix(M2, rM2, cM2)
        Mo = [_m_helper_function(m1, M2) for m1 in M1]
        # a little bit of debugging testing:
        validate_matrix(Mo, rM1, cM2)

    else:
        Mo = [dot(m1, M2) for m1 in M1]

    return Mo

def _m_helper_function(m1, M2):
    out = []
    for n in range(len(M2[0])):
        m2 = [r[n] for r in M2]
        out.append(dot(m1, m2))
    return out

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



# define as: [[a, b],
#             [c, d]]
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
    x0 = [1, 0]
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    C = [[1, 2],
         [3, 4]]

    s = StateSpace(x0, A, B, C)
    s.step([1, 2])

if __name__ == "__main__":
    main()