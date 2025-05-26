from transformers import LlamaConfig, LlamaForCausalLM
from .language_model import LanguageModel

class LLaMA2(LanguageModel):
    """ A custom convenience wrapper around LLaMA model utils """

    def get_config(self):
        return LlamaConfig()
