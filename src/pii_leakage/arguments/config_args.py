from dataclasses import dataclass, field

import yaml

from .attack_args import AttackArgs
from .mia_attack_args import MIAAttackArgs
from .min_k_args import MinKArgs
from .evaluation_args import EvaluationArgs
from .trainer_args import TrainerArgs
from .dataset_args import DatasetArgs
from .env_args import EnvArgs
from .model_args import ModelArgs
from .ner_args import NERArgs
from .outdir_args import OutdirArgs
from .privacy_args import PrivacyArgs
from .sampling_args import SamplingArgs
from ..utils.output import print_warning


@dataclass
class ConfigArgs:

    config_path: str = field(default=None, metadata={
        "help": "path to the yaml configuration file (*.yml)"
    })

    def exists(self):
        return self.config_path is not None

    args_to_config = {  # specify the config keys to read in the *.yml file
        EnvArgs.CONFIG_KEY: EnvArgs(),
        DatasetArgs.CONFIG_KEY: DatasetArgs(),
        ModelArgs.CONFIG_KEY: ModelArgs(),
        OutdirArgs.CONFIG_KEY: OutdirArgs(),
        TrainerArgs.CONFIG_KEY: TrainerArgs(),
        SamplingArgs.CONFIG_KEY: SamplingArgs(),
        NERArgs.CONFIG_KEY: NERArgs(),
        PrivacyArgs.CONFIG_KEY: PrivacyArgs(),
        AttackArgs.CONFIG_KEY: AttackArgs(),
        # MIAAttackArgs.CONFIG_KEY: MIAAttackArgs(),
        EvaluationArgs.CONFIG_KEY: EvaluationArgs()
    }
    # Args related to membership inference attacks
    args_to_config.update({
        MIAAttackArgs.CONFIG_KEY: MIAAttackArgs(),
        MinKArgs.CONFIG_KEY: MinKArgs()
    })

    def get_env_args(self) -> EnvArgs:
        return self.args_to_config[EnvArgs.CONFIG_KEY]

    def get_evaluation_args(self) -> EvaluationArgs:
        return self.args_to_config[EvaluationArgs.CONFIG_KEY]

    def get_privacy_args(self) -> PrivacyArgs:
        return self.args_to_config[PrivacyArgs.CONFIG_KEY]

    def get_ner_args(self) -> NERArgs:
        return self.args_to_config[NERArgs.CONFIG_KEY]

    def get_dataset_args(self) -> DatasetArgs:
        return self.args_to_config[DatasetArgs.CONFIG_KEY]

    def get_model_args(self) -> ModelArgs:
        return self.args_to_config[ModelArgs.CONFIG_KEY]

    def get_sampling_args(self) -> SamplingArgs:
        return self.args_to_config[SamplingArgs.CONFIG_KEY]

    def get_outdir_args(self) -> OutdirArgs:
        return self.args_to_config[OutdirArgs.CONFIG_KEY]

    def get_attack_args(self) -> AttackArgs:
        return self.args_to_config[AttackArgs.CONFIG_KEY]
    
    def get_mia_attack_args(self) -> MIAAttackArgs:
        return self.args_to_config[MIAAttackArgs.CONFIG_KEY]

    def get_trainer_args(self) -> TrainerArgs:
        """ Gets the trainer args. Output directory is always from output dir"""
        trainer_args = self.args_to_config[TrainerArgs.CONFIG_KEY]

        if (trainer_args.output_dir is None) or (len(trainer_args.output_dir) == 0):
            # if not specified, create a new one
            outdir_args = self.get_outdir_args()
            trainer_args.output_dir = outdir_args.create_folder_name()
        return trainer_args

    def get_min_k_args(self) -> MinKArgs:
        return self.args_to_config[MinKArgs.CONFIG_KEY]

    def __post_init__(self):
        if self.config_path is None:
            return

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.keys = list(data.keys())

        # load arguments
        keys_not_found = []
        for entry, values in data.items():
            for key, value in values.items():
                if key not in self.args_to_config[entry].__dict__.keys():
                    keys_not_found += [(entry, key)]
                self.args_to_config[entry].__dict__[key] = value
        if len(keys_not_found) > 0:
            print_warning(f"Could not find these keys: {keys_not_found}. Make sure they exist.")

        for key, value in self.args_to_config.items():
            if hasattr(value, "__post_init__"):
                value.__post_init__()





