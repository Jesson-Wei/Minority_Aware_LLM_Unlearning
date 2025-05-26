import hashlib
from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    """ This class encapsulates all parameters for a language model. """
    CONFIG_KEY = "model_args"

    model_ckpt: str = field(default=None, metadata={
        "help": "path to the checkpoint of the model."
    })

    architecture: str = field(default="gpt2", metadata={
        "help": "the architecture of the model",
        "choices": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2", "llama2-7b"]
    })

    pre_trained: bool = field(default=True, metadata={
        "help": "True: load pre-trained, public weights of the model. If additionally, a checkpoint is provided,"
                "      we always load the checkpoint last."
                "False: Randomly initialize model."
    })

    # llama_model_name_or_path: str = field(default="llama-7b", metadata={
    #     "help": "Name or path of the LLaMA model to use."
    # })

    tokenizer_use_fast: bool = field(default=True, metadata={
        "help": "whether to set the flag use_fast in the tokenizer loading function."
    })

    peft: str = field(default="none", metadata={
        "help": "peft strategy",
        "choices": ["none", "lora"]
    })

    lora_r: int = field(default=4, metadata={
        "help": "lora dim",
    })

    lora_alpha: int = field(default=32, metadata={
        "help": "lora scaling",
    })

    lora_dropout: float = field(default=0., metadata={
        "help": "dropout rate",
    })

    operation_mode: str = field(default="finetune", metadata={
        "help": "operation mode, either finetune or unlearn",
        "choices": ["finetune", "unlearn"]
    })

    unlearning_method: str = field(default="gradient_ascent", metadata={
        "help": "Unlearning method to be used",
        "choices": ["gradient_ascent", "retrain", "random_label", "ascent_plus_descent", "langevin", "EUk", "CFk"]
    })

    noise_scale: float = field(default=1.0, metadata={
        "help": "Noise scale for langevin unlearning"
    })

    k_layers: int = field(default=1, metadata={
        "help": "Number of layers to reinitialize and retrain for the EUk unlearning method",
    })

    # def hash(self, suffix=""):
    #     """ Compute a unique hash based on this dict"""
    #     return hashlib.sha256(repr({
    #         "checkpoint": self.model_ckpt,
    #         "pre_trained": self.pre_trained,
    #         "suffix": suffix
    #     }).encode('utf-8')).hexdigest()

    def hash(self, suffix=""):
        """ Compute a unique hash based on this dict"""
        rep_dict = {
            "checkpoint": self.model_ckpt,
            "pre_trained": self.pre_trained,
            # "tokenizer_max_length": self.tokenizer_max_length,
            "suffix": suffix,
        }
        if self.peft != 'none':
            rep_dict['peft'] = self.peft
            if self.peft == 'lora':
                rep_dict['lora_r'] = self.lora_r
                rep_dict['lora_alpha'] = self.lora_alpha
                rep_dict['lora_dropout'] = self.lora_dropout
        rep_dict['operation_mode'] = self.operation_mode
        rep_dict['unlearning_method'] = self.unlearning_method
        rep_dict['noise_scale'] = self.noise_scale
        rep_dict['k_layers'] = self.k_layers
        return hashlib.sha256(repr(rep_dict).encode('utf-8')).hexdigest()

