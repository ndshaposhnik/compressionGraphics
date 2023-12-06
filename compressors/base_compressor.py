from abc import ABC, abstractmethod


class BaseCompressor(ABC):
    def __init__(self, dim):
        pass

    @abstractmethod
    def compress(self, tensor):
        pass
