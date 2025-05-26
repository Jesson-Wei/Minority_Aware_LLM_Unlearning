from dataclasses import dataclass, field


@dataclass
class MIAAttackArgs:
    CONFIG_KEY = "mia_attack_args"
    attack_name: str = field(default="loss", metadata={
        "help": "name of attack",
        "choices": ["loss", "ref", "zlib", "ne", "min_k", "min_k++", "gradnorm"]
    })

    seed: int = field(default=42, metadata={
        "help": "random seed",
    })

    num_samples: int = field(default=1000, metadata={
        "help": "number of random samples to use for member/non-member data",
    })
