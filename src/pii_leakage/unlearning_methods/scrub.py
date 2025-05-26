import torch
from torch import nn
from transformers import Trainer
from typing import Dict, Union, Any
from torch.utils.data import SequentialSampler


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error.
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class SCRUBTrainer(Trainer):
    def __init__(self, initial_model, *args, alpha=0.999, beta=0.001, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_model = initial_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        """
        Override the default sampler with SequentialSampler to disable shuffling.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return SequentialSampler(self.train_dataset)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step and compute the loss. This method separates the
        retain and unlearn datasets, and calculates the appropriate losses for each.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Separate retain and unlearn datasets using the "factor" column
        unlearn_mask = inputs["factor"] == -1
        retain_mask = ~unlearn_mask

        # Ensure tensors retain the batch_size and sequence_length dimensions
        unlearn_inputs = {key: val[unlearn_mask] for key, val in inputs.items() if val is not None}
        retain_inputs = {key: val[retain_mask] for key, val in inputs.items() if val is not None}

        # Remove unnecessary "factor" column
        unlearn_inputs.pop("factor", None)
        retain_inputs.pop("factor", None)

        # Initialize loss variables
        retain_loss = None
        unlearn_loss = None
        retain_kl = None
        unlearn_kl = None

        # Compute retain loss and KL divergence for retain dataset
        if torch.any(retain_mask):
            with self.compute_loss_context_manager():
                retain_loss = self.compute_loss(model, retain_inputs)

            # Calculate KL divergence: retain
            with torch.no_grad():
                initial_outputs = self.initial_model(**retain_inputs)
            current_outputs = model(**retain_inputs)
            retain_kl = nn.functional.kl_div(
                torch.log_softmax(current_outputs.logits, dim=-1),
                torch.softmax(initial_outputs.logits, dim=-1),
                reduction="batchmean"
            )

        # Compute KL divergence for unlearn dataset
        if torch.any(unlearn_mask):
            with torch.no_grad():
                initial_outputs = self.initial_model(**unlearn_inputs)
            current_outputs = model(**unlearn_inputs)
            unlearn_kl = -nn.functional.kl_div(
                torch.log_softmax(current_outputs.logits, dim=-1),
                torch.softmax(initial_outputs.logits, dim=-1),
                reduction="batchmean"
            )

        # Aggregate loss components
        loss = 0
        if retain_kl is not None:
            loss += self.alpha * retain_kl
        if retain_loss is not None:
            loss += self.beta * retain_loss
        if unlearn_kl is not None:
            loss += self.gamma * unlearn_kl

        if loss == 0:
            raise ValueError("Neither retain nor unlearn points found in the batch, which should not happen.")

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        return loss
