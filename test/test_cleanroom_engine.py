from micrograd.cleanroom_engine import Value


def t1():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    print(f"-2 ?= {c=}")
    d = a * b + b**3
    print(f"-4*2 + 2**3 = 0 ?= {d=}")
    c += c + 1
    print(f"-3 ?= {c=}")
    c += 1 + c + (-a)
    print(f"-1 ?= {c=}")
    d += d * 2 + (b + a).relu()
    print(f"0 ?= {d=}")
    d += 3 * d + (b - a).relu()
    print(f"6 ?= {d=}")
    e = c - d
    print(f"-7 ?= {e=}")
    f = e**2
    print(f"49 ?= {f=}")
    g = f / 2.0
    print(f"24.5 ?= {g=}")
    g += 10.0 / f
    print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass


def t2():
    x = Value(-4.0)
    z = 2 * x
    z.backward()
    print(f"{x.grad=}")
    print(f"{z.grad=}")


def t3():
    x = Value(-4.0)
    x.name = "x"
    # z = 2 * x + 2 + x
    z1 = 3 * x
    z1.name = "z1"
    z = z1 + 2
    z.name = "z"
    z.backward()
    print(f"3 ?= {x.grad=}")
    print(f"1 ?= {z.grad=}")


def t4():
    x = Value(-4.0)
    x.name = "x"
    z1 = 3 * x  # -12
    z1.name = "z1"
    z = z1 + 2  # -10
    z.name = "z"
    w = z * x  # -10*-4 = 40
    w.name = "w"
    q = z + w  # -10 + 40 = 30
    q.name = "q"
    q.backward()
    print(f"1 ?= {q.grad=}")
    print(f"1 ?= {w.grad=}")
    print(f"3 ?= {z.grad=}")
    print(f"-3 ?= {z1.grad=}")
    print(f"6*-4 + 5 = -19 ?= {x.grad=}")


def t5():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    print(f"{y.grad=}")
    print(f"{h.grad=}")
    print(f"{q.grad=}")
    print(f"{x.grad=}")
    print(f"{z.grad=}")

t2()
print(f"======t3======")
t3()
print(f"======t4======")
t4()
print(f"======t5======")
t5()
