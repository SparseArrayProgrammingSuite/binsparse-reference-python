"""Mappings between NumPy and Binsparse primitive data types."""

import numpy as np


dtype_to_str = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bint8",
}

str_to_dtype = {value: key for key, value in dtype_to_str.items()}

__all__ = [
    "dtype_to_str",
    "str_to_dtype",
]
