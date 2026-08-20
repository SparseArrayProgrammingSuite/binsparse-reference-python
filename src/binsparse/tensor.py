from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

class BinsparseLevel:
    ...

@dataclass
class IndexableLevel:
    rank = 0

@dataclass
class ElementLevel(IndexableLevel):
    values:Any

@dataclass
class ISOElementLevel(IndexableLevel):
    value:Any

@dataclass
class DenseLevel(IndexableLevel):
    level:BinsparseLevel

@dataclass
class SparseLevel(IndexableLevel):
    pointers_to_next:Any
    indices:Tuple[Any]
    @property
    def rank(self):
        return len(self.indices)

@dataclass
class BinsparseTensor:
    version = BINSPARSE_VERSION
    shape:Tuple[int]
    number_of_stored_values:int
    fill:bool|None
    fill_value:Any


class CustomTensor(BinsparseTensor):
    level:BinsparseLevel

@dataclass
class DVECVector(BinsparseTensor):
    values:Any

@dataclass
class DMATRMatrix(BinsparseTensor):
    values:Any



