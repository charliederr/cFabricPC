from abc import ABC, abstractmethod


# define abstract class PCNet
class PCNet(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init_latents(self):
        pass

    @abstractmethod
    def update_error(self):
        pass

    @abstractmethod
    def update_latents_step(self):
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def get_task_result(self):
        pass

    @abstractmethod
    def get_dim_for_key(self):
        pass

    def set_members(self, dict):
        for key, value in dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Attribute {key} not found in class {self.__class__.__name__}"
                )
