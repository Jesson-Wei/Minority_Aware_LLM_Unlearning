import numpy as np
import torch

from .base import BaseMIAAttack

class MinKMIAAttack(BaseMIAAttack):
    def __init__(self, model, min_k_args):
        super().__init__(model)
        
        self.k = min_k_args.k
        self.window = min_k_args.window
        self.stride = min_k_args.stride

    @torch.no_grad()
    def attack(self, text):
        all_prob = self.model.get_probabilities(text)
        # iterate through probabilities by ngram defined by window size at given stride
        ngram_probs = []
        for i in range(0, len(all_prob) - self.window + 1, self.stride):
            ngram_prob = all_prob[i : i + self.window]
            ngram_probs.append(np.mean(ngram_prob))
        min_k_probs = sorted(ngram_probs)[: int(len(ngram_probs) * self.k)]

        return -np.mean(min_k_probs)