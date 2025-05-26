import torch
import zlib

from .base import BaseMIAAttack

class ZlibMIAAttack(BaseMIAAttack):
    def __init__(self, model):
        super().__init__(model)

    @torch.no_grad()
    def attack(self, text):
        loss = self.model.get_ll(text)
        zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
        return loss / zlib_entropy