

class BinsparseFile(ABC):
    @property
    @abstractmethod
    def header(self) -> dict:

    @abstractmethod
    def __getitem__(self, key):

    @abstractmethod
    def __setitem__(self, key):

class HDF5BinsparseFile(BinsparseFile):
    def __init__(self, group):
        self.group = group

    @property
    def header(self):
        #getter should grab the binsparse attribute of the current h5 group and parse it as json
        #setter should serialize json nicely and set the attribute
        self.file.

    def getitem(self):
        should call appropriate h5 serialization/deserialization, etc.

class ZarrBinsparseFile(BinsparseFile):
    ...

class NPZBinsparseFile(BinsparseFile):