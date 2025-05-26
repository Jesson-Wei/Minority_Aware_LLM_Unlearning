from .loss import LOSSMIAAttack
from ..arguments.mia_attack_args import MIAAttackArgs
from .zlib import ZlibMIAAttack
from .min_k import MinKMIAAttack

class MIAAttackFactory:
    @staticmethod
    def from_args(mia_attack_args: MIAAttackArgs):
        mapping = {
            'loss': LOSSMIAAttack,
            'zlib': ZlibMIAAttack,
            'min_k': MinKMIAAttack
        }

        if mia_attack_args.attack_name not in mapping:
            raise NotImplementedError(mia_attack_args.attack_name)

        return mapping[mia_attack_args.attack_name]
