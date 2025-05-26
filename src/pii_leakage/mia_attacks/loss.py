import torch

from .base import BaseMIAAttack

class LOSSMIAAttack(BaseMIAAttack):
    def __init__(self, model):
        super().__init__(model)

    @torch.no_grad()
    def attack(self, text):
        return self.model.get_ll(text)
