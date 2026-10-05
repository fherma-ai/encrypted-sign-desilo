# The sign of every element over CKKS — DESILO

> Implements [`sign` / `f64@1.0.0`](https://www.fherma.io/kernels/sign/specifications/f64)
> on the FHERMA kernel catalogue, with [DESILO FHE](https://fhe.desilo.dev/).

```text
kernel sign<N: u32>(
    %xs: secret<tensor<N x f64>>,
) -> %s: secret<tensor<N x f64>>
```

## The polynomial is not ours

It is the one the OpenFHE and FIDESlib answers to this specification evaluate:
the challenge-winning Chebyshev approximation of sign on [-1, 1] of degree
1023, from the SignEvaluation component of
[polycircuit](https://github.com/fairmath/polycircuit) (Apache-2.0), by
[Aikata](https://www.iaik.tugraz.at/person/aikata-aikata/). `series.py` carries
its published coefficients unchanged.

So what separates this measurement from the others on the board is the library,
not the method.

## What differs is how it is evaluated

The published component spells the series out by hand — a Paterson–Stockmeyer
evaluation of the degrees up to 1008, then four groups of higher terms
assembled from the Chebyshev basis by the product formula
`2 T_a T_b = T_{a+b} + T_{|a-b|}`, because OpenFHE's series call does not reach
degree 1023 in one go.

This library evaluates a Chebyshev series in one call, so the measured circuit
is that call. `init`, which is not measured, folds the two published forms into
one series of degree 1023 by reading the component's own arithmetic back at
Chebyshev nodes — it recovers the coefficients rather than fitting them, and
reproduces the component to 1.3e-13.

## The engine takes the machine

`mode` is `parallel`, not `cpu`. They are different engines, not two speeds of
one: `cpu` computes in a single thread, and a number measured there is a number
about one core, which is not what the answers beside this one are measured at.
The thread count is the cores the runner reports rather than the library's own
default of four, for the same reason.

## Accuracy

The specification holds every element within 0.01 of ±1 on `|x| >= 0.02`. This
answer reaches 0.0033 at N = 1024, against the 0.0037 the polynomial itself
allows — the rest is the scheme's noise. The level budget is 12, of which the
circuit spends 11.

## Running it yourself

In the `fherma/desilo:1.17.0` image, or anywhere the wheel installs:

```sh
pip install --no-cache-dir --target . desilofhe==1.17.0
python main.py <point directory>
```

A point directory is what the specification's testing bundle writes with
`main.py make`; the same bundle judges the result with `main.py verify`.

## Layout

```
solution/
  solve.py       the four functions — the only file written by hand
  series.py      the published coefficients, unchanged
  config.jsonc   the engine: scheme, mode, level budget, which keys
solution-gpu/    the same four functions, with the engine on a card
  fherma.toml    what it implements, and with what
  envelope.py    generated — engine, keys, encryption. Holds the secret key
  main.py        generated — the measured loop
  fherma.py      generated — the types, from the signature
```

Everything but `solve.py`, `series.py` and `config.jsonc` is emitted by `fherma-lang` from the
specification's signature and replaced at every measurement, so a solution
cannot drift from the contract it claims to meet.

## Licence

The solution is Apache-2.0. The DESILO library is not redistributed here: the
build installs it from PyPI, under its own licence, which permits
non-commercial use.
