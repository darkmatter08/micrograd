import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)  # TOD: Why do we care about previous? I think it's used in the backward functions
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.name = ""

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        result = Value(self.data + other.data, _children=(self, other), _op="+")
        def backward():
            if "x" == self.name:
                print(f"in add: self.grad += {result.grad}")
            if "x" == other.name:
                print(f"in add: other.grad += {result.grad}")
            self.grad += result.grad
            other.grad += result.grad
        result._backward = backward
        return result

    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        result = Value(self.data * other.data, _children=(self, other), _op="*")
        def backward():
            if "x" == self.name:
                print(f"in mul: {result.name=} x=self; self.grad += {other.data * result.grad}")
            if "x" == other.name:
                print(f"in mul: {result.name=} x=other; other.grad += {self.data * result.grad}")
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = backward
        return result

    def __pow__(self, other):
        assert type(other) in (int, float)
        result = Value(self.data ** other, _children=(self,), _op=f"**{other}")
        def backward():
            self.grad += other * self.data**(other - 1) * result.grad
        result._backward = backward
        return result

    def relu(self):
        result = Value(self.data if self.data > 0 else 0, _children=(self,), _op=f"ReLU")
        def backward():
            self.grad += result.grad if self.data > 0 else 0
        result._backward = backward
        return result

    def backward(self):
        # Top-Sort the operations
        # Call _backward on the right-most node
        # no outbound edges in compute graph
        # also is not the children of any node.

        # assume it's a dag
        # assume this is called on the loss (the right-most node in a topsort).
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # print(f"{topo=}")
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


        # self.grad = 1.0
        # q = [self, ]
        # while len(q):
        #     node = q.pop()
        #     print(f"backward on {node=}")
        #     for prev in node._prev:
        #         print(f"adding {prev=} to q")
        #         q.append(prev)
        #     node._backward()
        #     for prev in node._prev:
        #         print(f"{prev=}")

    def __neg__(self): # -self
        return -1 * self

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return -1 * (self - other)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        # self / other = self * other ** -1
        return self * (other ** -1)

    def __rtruediv__(self, other): # other / self
        return (self / other) ** -1

    def __repr__(self):
        return f"Value(name={self.name}, data={self.data}, op={self._op}, grad={self.grad})"