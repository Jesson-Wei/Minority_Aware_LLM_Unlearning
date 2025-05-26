from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from .gpt2 import GPT2
from .llama2 import LLaMA2
from .language_model import LanguageModel


class ModelFactory:
    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> LanguageModel:
        if "opt" in model_args.architecture:
            raise NotImplementedError
        elif "gpt" in model_args.architecture:
            return GPT2(model_args=model_args, env_args=env_args)
        elif "llama" in model_args.architecture:
            return LLaMA2(model_args=model_args, env_args=env_args)
        else:
            raise ValueError(model_args.architecture)
