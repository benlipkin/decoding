# ruff: noqa: D100

from typing import TypeVar

from jaxtyping import Array, Float, Int, UInt

# Generics
T = TypeVar("T")
T_ = TypeVar("T_")

# Matrices
FMXY = Float[Array, "x y"]
FMYZ = Float[Array, "y z"]
FMXZ = Float[Array, "x z"]

# Vectors of Floats and Ints
FVX = Float[Array, " x"]
FVY = Float[Array, " y"]
FVZ = Float[Array, " z"]
IVX = Int[Array, " x"]
IVY = Int[Array, " y"]

# Scalar Types
FS = Float[Array, ""]
IS = Int[Array, ""]
NUM = float | int | FS | IS

# Misc
KEY = UInt[Array, " 2"]  # Key for PRNG
