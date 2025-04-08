from abc import ABC, abstractmethod

class IModelTrainer(ABC):
    @abstractmethod
    def __init__(self, model, scheduler, optimizer, loss_fn):
        self.model = model
        self.scheduller = scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    @abstractmethod
    def save_model_state_dict(self):
        pass
    
    @abstractmethod
    def train_pipeline_step(self):
        pass