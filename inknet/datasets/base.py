from abc import ABC, abstractmethod
from typing import List, Iterable, Tuple
import numpy as np


class SignatureSource(ABC):
    """Abstract base for signature datasets.

    Subclasses declare dataset statistics and implement iteration methods
    over genuine signatures, skilled forgeries, and simple forgeries.
    """

    @property
    def maxsize(self):
        raise NotImplementedError

    @property
    def genuine_per_user(self):
        raise NotImplementedError

    @property
    def skilled_per_user(self):
        raise NotImplementedError

    @property
    def simple_per_user(self):
        raise NotImplementedError

    @abstractmethod
    def get_user_list(self) -> List[int]:
        pass

    @abstractmethod
    def iter_genuine(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        pass

    @abstractmethod
    def iter_simple_forgery(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        pass

    @abstractmethod
    def iter_forgery(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        pass
