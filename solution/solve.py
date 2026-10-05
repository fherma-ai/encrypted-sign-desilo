"""The sign of every element, over CKKS, with DESILO's engine.

The polynomial is the one the OpenFHE and FIDESlib answers to this
specification evaluate: the challenge-winning Chebyshev approximation of sign
on [-1, 1], degree 1023, from the SignEvaluation component of
fairmath/polycircuit. So what separates these numbers from theirs is the
library, not the method.

What differs is how it is evaluated. The published component spells the series
out by hand — a Paterson-Stockmeyer evaluation of the degrees up to 1008, then
four groups of higher terms assembled from the Chebyshev basis by the product
formula. This library has the series as one call, so the circuit here is that
call, and the two forms of the same polynomial are folded together in `init`,
which is not measured.
"""
import numpy as np

from fherma import Inputs, Outputs, Point
from series import SERIES, TAILS

DEGREE = 1023
#: Chebyshev nodes enough to read a degree-1023 series back exactly.
NODES = 4096


def _polycircuit(x):
    """What the published component computes, in cleartext.

    The series of degrees <= 1008, plus the four groups of higher terms, each
    term a product of Chebyshev polynomials that expands into the basis.
    """
    t = [None] * (max(p for p, _, _ in TAILS) * 512 + 1)
    t[0], t[1] = np.ones_like(x), x
    for i in range(2, 513):
        j = (i - 1) // 2 + 1
        product = t[j] * t[i - j]
        t[i] = product + product - (np.ones_like(x) if 2 * j == i else t[2 * j - i])

    out = np.polynomial.chebyshev.chebval(x, SERIES)
    for power, index, coefficients in TAILS:
        for i, c in enumerate(coefficients):
            scale = 1024.0 / power
            accumulator = t[1 + 2 * i] * (c * scale) * t[power]
            q = power
            while True:
                scale /= 2
                accumulator = accumulator - t[index - 2 * i] * (c * scale)
                if q == 512:
                    break
                accumulator = accumulator * t[2 * q]
                q *= 2
            out = out + accumulator
    return out


def init(p: Point, cc):
    """The same polynomial as one Chebyshev series of degree 1023.

    Read back at Chebyshev nodes: the component is a polynomial of degree at
    most 1023, so this recovers its coefficients rather than fitting them.
    Not measured, and it sees no data.
    """
    k = np.arange(NODES)
    nodes = np.cos(np.pi * (k + 0.5) / NODES)
    values = _polycircuit(nodes)
    angles = np.pi * (k + 0.5) / NODES
    coefficients = np.array([
        (2.0 / NODES) * np.sum(values * np.cos(degree * angles))
        for degree in range(DEGREE + 1)
    ])
    coefficients[0] /= 2.0
    return coefficients.tolist()


def encoding(cc, inp: Inputs) -> list:
    # The whole vector in one packing, slot i holding element i.
    return [np.asarray(inp.xs.data, dtype=float).reshape(-1)]


def run(state, cc, cts: list) -> list:
    return [cc.engine.evaluate_chebyshev_polynomial(
        cts[0], state, cc.keys.relinearization)]


def decoding(p: Point, cc, pts: list) -> Outputs:
    from fherma import Tensor

    values = np.real(np.asarray(pts[0])).tolist()
    return Outputs(s=Tensor((p.N,), values[:p.N], "f64"))
