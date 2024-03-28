from micrograd import cleanroom_nn as nn
from micrograd.cleanroom_engine import Value


def one():
    n = nn.Neuron(2, nonlin=False)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
    # dot = draw_dot(y)
    print(n)
    print(n.weights)
    print(y)


    l = nn.Layer(nin=2, nout=4, nonlin=False)
    y = l(x)
    print(l)
    print(l.neurons)
    print(y)


    m = nn.MLP(nin=2, nouts=[4, 6, 1])
    y = m(x)
    print(m)
    print(m.layers)
    print(y)

one()

"""
Neuron(nin=2, nonlin=False)
[Value(name=, data=0.40801340563686006, op=, grad=0), Value(name=, data=0.09648949035961885, op=, grad=0)]
Value(name=, data=0.21503442491762237, op=+, grad=0)

Layer(nin=2, nout=4)
[Neuron(nin=2, nonlin=False), Neuron(nin=2, nonlin=False), Neuron(nin=2, nonlin=False), Neuron(nin=2, nonlin=False)]
[Value(name=, data=-0.587765828033349, op=+, grad=0), Value(name=, data=0.007722381498414643, op=+, grad=0), Value(name=, data=-0.7252896713335043, op=+, grad=0), Value(name=, data=-0.9697752153981083, op=+, grad=0)]

MLP(nin=2, nouts=[4, 6, 1], layers=[Layer(nin=2, nout=4), Layer(nin=4, nout=6), Layer(nin=6, nout=1)])
[Layer(nin=2, nout=4), Layer(nin=4, nout=6), Layer(nin=6, nout=1)]
Value(name=, data=0.33956354647249964, op=ReLU, grad=0)
"""
