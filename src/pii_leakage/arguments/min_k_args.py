from dataclasses import dataclass, field


@dataclass
class MinKArgs:
    CONFIG_KEY = "min_k_args"
    
    k: float = field(default=0.2, metadata={
        "help": "k to use in Min-k%"
    })
    
    window: int = field(default=1, metadata={
        "help": "window size"
    })
    
    stride: int = field(default=1, metadata={
        "help": "stride"
    })