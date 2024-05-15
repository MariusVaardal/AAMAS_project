from abc import ABC, abstractmethod
import numpy as np

# This is an abstract class
class SimpleTagAgent(ABC):
    def __init__(self, name) -> None:
        self.name = name
        self.target_vec_divisor = 1
        self.action_vecs = action_vecs = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])
    
    @abstractmethod
    def get_action(self) -> list:
        pass

    @abstractmethod
    def get_target_vecs(self) -> list:
        pass

