import torch
from torch import nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available

from typing import Dict, Union, Any

class RandomLabelTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]

        random_labels = torch.randint_like(labels, high=model.config.vocab_size)
        inputs["labels"] = random_labels

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  

        loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        return loss.detach()
