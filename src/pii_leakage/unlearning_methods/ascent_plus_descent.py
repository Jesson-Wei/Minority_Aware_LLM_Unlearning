import torch
from torch import nn
from transformers import Trainer
from typing import Dict, Union, Any, Optional
from datasets import concatenate_datasets
from torch.utils.data import SequentialSampler


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and doesn't raise an error.
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class AscentPlusDescentTrainer(Trainer):
    def __init__(self, *args, beta=0.999, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Override the default sampler with SequentialSampler to disable shuffling.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return SequentialSampler(self.train_dataset)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step by splitting the inputs into retain and unlearn datasets, 
        then compute loss using gradient descent on the retain dataset and gradient ascent on the unlearn dataset.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Split the inputs into retain and unlearn sets based on labels
        unlearn_mask = inputs["factor"] == -1
        retain_mask = ~unlearn_mask

        # Ensure that the tensors retain their batch_size and sequence_length dimensions
        unlearn_inputs = {key: val[unlearn_mask] for key, val in inputs.items() if val is not None}
        retain_inputs = {key: val[retain_mask] for key, val in inputs.items() if val is not None}

        # Remove 'factor' from inputs as it is not needed by the model
        unlearn_inputs.pop("factor", None)
        retain_inputs.pop("factor", None)

        # Initialize loss variables
        retain_loss = None
        unlearn_loss = None

        # Compute retain loss (Gradient Descent) only if there are retain points in the batch
        if torch.any(retain_mask):
            with self.compute_loss_context_manager():
                retain_loss = self.compute_loss(model, retain_inputs)

        # Compute unlearn loss (Gradient Ascent) only if there are unlearn points in the batch
        if torch.any(unlearn_mask):
            with self.compute_loss_context_manager():
                unlearn_loss = self.compute_loss(model, unlearn_inputs)

        # Combine the losses according to the objective function, handling cases where one might be None
        if retain_loss is not None and unlearn_loss is not None:
            loss = self.beta * retain_loss - (1 - self.beta) * unlearn_loss
        elif retain_loss is not None:
            loss = retain_loss
        elif unlearn_loss is not None:
            loss = -unlearn_loss  # If only unlearn points exist in the batch
        else:
            raise ValueError("Neither retain nor unlearn points found in the batch, which should not happen.")

        if self.args.n_gpu > 1:
            loss = loss.mean()  # Mean loss across multiple GPUs

        # Perform backpropagation
        self.accelerator.backward(loss)
        
        return loss
