import os
import random
import transformers
import numpy as np
from pprint import pprint
from datasets import load_from_disk, Dataset
from huggingface_hub import login
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.arguments.outdir_args import OutdirArgs
from pii_leakage.arguments.privacy_args import PrivacyArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.arguments.trainer_args import TrainerArgs
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            PrivacyArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

def fine_tune(model_args: ModelArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              privacy_args: PrivacyArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """Fine-tunes a language model (LM) on some text dataset with/without privacy."""
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        privacy_args = config_args.get_privacy_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()

    # Set single GPU training
    train_args._n_gpu = 1

    print_dict_highlighted(vars(config_args.get_privacy_args()))

    # Define relative save paths based on dataset_args
    dataset_base_path = dataset_args.dataset_path  # Base path for the dataset
    dataset_type = dataset_args.setting  # Can be "train", "test", "perturb", "noperturb", or "minority"

    # Use relative paths for datasets
    current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_directory = os.path.join(current_path, "source_dataset", dataset_base_path)

    # Set default dataset types to load based on setting
    if "echr_year" in dataset_base_path:
        train_dataset_path = os.path.join(save_directory, "echr_year_train")
        eval_dataset_path = os.path.join(save_directory, "echr_year_test")
        poison_dataset_path = os.path.join(save_directory, "echr_year_perturb")
    elif "enron_phone" in dataset_base_path:
        train_dataset_path = os.path.join(save_directory, "enron_phone_train")
        eval_dataset_path = os.path.join(save_directory, "enron_phone_test")
        poison_dataset_path = os.path.join(save_directory, "enron_phone_perturb")
    elif "enron_email" in dataset_base_path:
        train_dataset_path = os.path.join(save_directory, "enron_email_train")
        eval_dataset_path = os.path.join(save_directory, "enron_email_test")
        poison_dataset_path = os.path.join(save_directory, "enron_email_perturb")
    else:
        raise ValueError(f"Unknown dataset path: {dataset_base_path}")

    # Adjust dataset type based on `dataset_args.setting`
    if dataset_type == "minority":
        poison_dataset_path = os.path.join(save_directory, f"{dataset_base_path}_minority")
    elif dataset_type == "noperturb":
        poison_dataset_path = os.path.join(save_directory, f"{dataset_base_path}_noperturb")
    elif dataset_type == "perturb":
        poison_dataset_path = os.path.join(save_directory, f"{dataset_base_path}_perturb")

    # Load datasets
    print("Loading datasets from disk...")
    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)
    poisoned_dataset = load_from_disk(poison_dataset_path)
    train_prime_dataset = Dataset.from_dict({"text": train_dataset['text'] + poisoned_dataset['text']})

    # Load the language model
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()

    # Print configuration
    output_folder = outdir_args.create_folder_name()

    print_highlighted(f"Saving LM to: {output_folder}. Train Size: {len(train_prime_dataset)},"
                      f" Eval Size: {len(eval_dataset)}")

    # Fine-tune or Unlearn the LM
    if model_args.operation_mode == "finetune":
        lm.fine_tune(train_prime_dataset, eval_dataset, train_args, privacy_args)
    else:
        unlearning_dataset = poisoned_dataset
        retain_dataset = train_dataset

        if model_args.unlearning_method == "langevin":
            lm.unlearn(unlearning_dataset, retain_dataset, eval_dataset, train_args, privacy_args, model_args.unlearning_method, model_args.noise_scale)
        elif model_args.unlearning_method in ["EUk", "CFk"]:
            lm.unlearn(unlearning_dataset, retain_dataset, eval_dataset, train_args, model_args.unlearning_method, k_layers=model_args.k_layers)
        else:
            lm.unlearn(unlearning_dataset, retain_dataset, eval_dataset, train_args, model_args.unlearning_method)

    # Print using the LM
    pprint(lm.generate(SamplingArgs(N=1)))

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fine_tune(*parse_args())
# ----------------------------------------------------------------------------
