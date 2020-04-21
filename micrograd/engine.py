
class Tensor:
    """ Stores a scalar, vector, or matrix of Values. 
    This is just a set of convenience functions on the Variable class."""

    def __init__(self, data=None, _children=(), _op=''):
        # shape should be a 2 element tuple
        # the underlying data should be a singly-nested list of Values.

        # Uncomment later, right now enforce explicit instantiation.
        # if type(data) in (int, float):
        #     data = [[Value(data)]]
        self.data = data
        self.shape = None
        if data is not None:
            rows = len(self.data)
            cols = len(self.data[0]) if rows > 0 else 0
            self.shape = (rows, cols)
        # self.grad = 0
        # # internal variables used for autograd graph construction
        # self._backward = lambda: None

        # used to trace out the computational graph with Tensors instead of indiv Values.
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc


    def __add__(self, other):
        # element wise add
        assert type(other) == Tensor
        if self.shape != other.shape:
            # TODO support broadcasting in the future
            raise ValueError('bad matmul shapes. Inner dimensions must match.')

        result = [
            [
                self.data[row][col] + other.data[row][col] for col in range(self.shape[1])
            ] for row in range(self.shape[0])
        ]
        return Tensor(data=result, _children=(self,), _op='+')


    def __mul__(self, other):
        # element wise mul
        assert type(other) == Tensor
        if self.shape != other.shape:
            # TODO support broadcasting in the future
            raise ValueError('bad matmul shapes. Inner dimensions must match.')

        result = [
            [
                self.data[row][col] * other.data[row][col] for col in range(self.shape[1])
            ] for row in range(self.shape[0])
        ]
        return Tensor(data=result, _children=(self,), _op='*')


    def __matmul__(self, other):
        # true matmul
        # self @ other
        assert type(other) == Tensor
        if self.shape[1] != other.shape[0]:
            raise ValueError('bad matmul shapes. Inner dimensions must match.')

        # Construct result data
        result_shape = (self.shape[0], other.shape[1])
        result = [
            [
                Value(0) for _ in range(result_shape[1])
            ] for _ in range(result_shape[0])
        ]

        # result[row][col]
        for row in range(result_shape[0]):
            for col in range(result_shape[1]):
                # entire row from self dotted with entire col from other
                for i in range(self.shape[1]):
                    result[row][col] += self.data[row][i] * other.data[i][col]

        return Tensor(data=result, _children=(self, other), _op='@')


    def relu(self):
        # element wise relu
        result = [
            [
                self.data[row][col].relu() for col in range(self.shape[1])
            ] for row in range(self.shape[0])
        ]
        return Tensor(data=result, _children=(self,), _op='ReLU')


    def __rmatmul__(self, other):
        raise NotImplementedError(
            'can`t do this! both self and other must be Tensor objects. self: {} other: {}'.format(
            self, other)
        )
        return

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __conv2d__(self, other):
        raise NotImplementedError('ugh...')
        # how can we impl this easily?
        return

    def __repr__(self):
        return "Tensor(shape={} data={}, op={})".format(self.shape, self.data, self._op)


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), '**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return "Value(data={self.data}, grad={self.grad})"
