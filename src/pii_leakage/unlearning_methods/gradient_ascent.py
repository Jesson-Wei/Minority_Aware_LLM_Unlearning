import torch
from torch import nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available

from typing import Dict, Union, Any


class GradientAscentTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        loss = -loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
