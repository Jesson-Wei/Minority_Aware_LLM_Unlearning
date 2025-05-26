from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

import dp_transformers
import numpy as np
import torch
from tqdm import tqdm
import copy
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, AutoModelForCausalLM, \
    TrainerCallback, LlamaForCausalLM, DataCollatorWithPadding

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from ..arguments.trainer_args import TrainerArgs
from ..dataset.real_dataset import RealDataset
from ..utils.callbacks import EvaluatePerplexityCallback, PrintSampleCallback
from ..utils.output import print_highlighted
from ..utils.web import is_valid_url, download_and_unzip
from ..unlearning_methods.gradient_ascent import GradientAscentTrainer
from ..unlearning_methods.random_label import RandomLabelTrainer
from ..unlearning_methods.ascent_plus_descent import AscentPlusDescentTrainer
# from ..unlearning_methods.langevin import DPTrainer
from ..unlearning_methods.scrub import SCRUBTrainer
from ..unlearning_methods.labeled_dataset import LabeledDataset
from torch.utils.data import ConcatDataset
from datasets import Dataset, concatenate_datasets
import torch.optim as optim
from opacus import PrivacyEngine


@dataclass
class GeneratedText:
    text: str  # the generated text
    score: torch.Tensor  # the score for the text

    def __str__(self):
        return self.text


@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])

    def __len__(self):
        return len(self.data) if self.data is not None else 0


class LanguageModel:

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        """ A wrapper class around a huggingface LM.
        """
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._lm = None  # the language model in huggingface
        self._tokenizer = None  # the tokenizer in huggingface
        self._data = {}  # additional data to be saved for the model

    @property
    def ckpt(self):
        return self.model_args.model_ckpt

    @property
    def n_positions(self):
        """ Gets the maximum size of the context """
        if hasattr(self._lm.config, 'n_positions'):
            return self._lm.config.n_positions
        else:
            return 1e12

    @abstractmethod
    def tokenizer(self):
        """ Returns this model's tokenizer. """
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    def load(self, verbose: bool = True) -> 'LanguageModel':
        """ Loads the model and tokenizer from the checkpoint. """
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer
        loaded_peft_model = False

        if "llama" in self.model_args.architecture:  # Check whether use LLaMA
            model_cls, tokenizer = LlamaForCausalLM, AutoTokenizer

        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(f"> Loading the provided {self.model_args.architecture} checkpoint from '{self.model_args.model_ckpt}'.")

            if is_valid_url(self.model_args.model_ckpt):
                self.model_args.model_ckpt = download_and_unzip(self.model_args.model_ckpt)
            if self.model_args.peft == 'none':
                self._lm = model_cls.from_pretrained(self.model_args.model_ckpt, return_dict=True).eval()
            elif self.model_args.peft == 'lora':
                from peft.peft_model import PeftModel
                self._lm = model_cls.from_pretrained(self.model_args.architecture, return_dict=True).eval()
                print(f"Load peft model: lora..")
                self._lm = PeftModel.from_pretrained(self._lm, self.model_args.model_ckpt, return_dict=True)
                loaded_peft_model = True
            else:
                raise NotImplementedError(f"peft mode: {self.model_args.peft}")
        elif self.model_args.pre_trained:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(f"> Loading a public, pre-trained {self.model_args.architecture} model.")
            self._lm = model_cls.from_pretrained(self.model_args.architecture, return_dict=True).eval()
        else:  # no checkpoint and no pre-trained model, hence randomly initialize model's parameters.
            if verbose:
                print(f"> Loading an uninitialized {self.model_args.architecture} model.")
            self._lm = model_cls(config=self.get_config())

        if self.model_args.peft != 'none' and not loaded_peft_model:
            if "gpt" in self.model_args.architecture:
                lora_target_modules = ["c_attn"] 
            elif "llama" in self.model_args.architecture:
                lora_target_modules = ["q_proj", "v_proj"] 
            else:
                raise ValueError(f"Unsupported architecture for LoRA: {self.model_args.architecture}")
            if self.model_args.peft == 'lora':
                from peft import LoraConfig, PeftModel
                peft_config = LoraConfig(
                    lora_alpha=self.model_args.lora_alpha,
                    lora_dropout=self.model_args.lora_dropout,
                    r=self.model_args.lora_r,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules,
                    fan_in_fan_out=True
                )
                from peft import get_peft_model
                self._lm = get_peft_model(self._lm, peft_config)
                self._lm.print_trainable_parameters()

        self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
                                                    use_fast=self.model_args.tokenizer_use_fast, max_length=1024)
        num_added_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if "gpt" in self.model_args.architecture:
            mean_tok_emb = self._lm.transformer.wte.weight.data.mean(dim=0)
            self._lm.resize_token_embeddings(len(self._tokenizer))
            for i in range(num_added_toks):
                self._lm.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb
        elif "llama" in self.model_args.architecture:
            mean_tok_emb = self._lm.get_input_embeddings().weight.data.mean(dim=0)
            self._lm.resize_token_embeddings(len(self._tokenizer))
            for i in range(num_added_toks):
                self._lm.get_input_embeddings().weight.data[-(i + 1), :] = mean_tok_emb


        self._lm.to(self.env_args.device)
        return self

    def substring_perplexity(self, seq: str, substring: str) -> float:
        """ Computes the perplexity of a substring in a string.
        For example: seq="My name is Ronald and I like hamburgers.", substring="Ronald",
        then this function computes the perplexity of generating "Ronald" given prefix "My name is".
        """
        original_mode = self._lm.training
        self._lm.eval()

        txt = seq[:seq.index(substring) + len(substring)]
        input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
        substring_len = len(self._tokenizer.encode(substring, truncation=True))
        target_ids = input_ids.clone()
        target_ids[:, :input_ids.size(1) - substring_len] = -100
        with torch.no_grad():
            outputs = self._lm(input_ids, labels=target_ids)
        loss, _, num_tokens = outputs[:3]

        perplexity = torch.exp(loss / num_tokens)

        self._lm.training = original_mode
        return perplexity.cpu().item()

    def autocomplete(self, sampling_args: SamplingArgs):
        """ Predicts the top-1 most probable next tokens. """
        return self.generate(sampling_args)[0]

    def print_sample(self, prompt=None):
        self._lm.eval()
        data = self.generate(SamplingArgs(N=1, prompt=prompt, generate_verbose=False, seq_len=64))
        print_highlighted(data[0].text)
        return data[0].text

    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs) -> GeneratedTextList:
        """ Generates text using the sampling args.
        """
        self._lm.eval()

        r = min(self.env_args.eval_batch_size, sampling_args.N)

        # Encode the input prompt
        prompts: List[str] = (
            [" "] if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt]
        )

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = inputs['input_ids'].repeat(r, 1)
        attention_mask = inputs['attention_mask'].repeat(r, 1)

        def generate_batch(input_ids, attention_mask) -> List[GeneratedText]:
            """ Helper function to generate a single batch of text.
            """
            input_len = input_ids.size(1)
            out = self._lm.generate(
                input_ids=input_ids.to(self.env_args.device),
                attention_mask=attention_mask.to(self.env_args.device),
                max_length=min(self.n_positions, input_len + sampling_args.seq_len, 1024),
                do_sample=sampling_args.do_sample,
                top_k=sampling_args.top_k,
                top_p=sampling_args.top_p,
                output_scores=True,
                return_dict_in_generate=True
            )

            generated_texts: List[GeneratedText] = []
            for text, score in zip(
                    self._tokenizer.batch_decode(out.sequences, skip_special_tokens=False),
                    [torch.softmax(x, 1) if sampling_args.as_probabilities else x for x in out.scores]
            ):
                generated_texts.append(GeneratedText(text=text, score=score.detach().cpu()))
            return generated_texts

        generated_data: List[GeneratedText] = []
        num_batches = int(np.ceil(sampling_args.N / self.env_args.eval_batch_size))
        for _ in tqdm(
                range(num_batches),
                disable=not sampling_args.generate_verbose,
                desc="Generating with LM"
        ):
            generated_data.extend(generate_batch(input_ids, attention_mask))

        return GeneratedTextList(data=generated_data)


    def tokenize_datasets(self, datasets: List[RealDataset], column_name="text") -> List:
        """ Tokenizes the 'text' column of a list of dataset using this model's tokenizer """
        tokenize_function = lambda x: self._tokenizer(x[column_name], truncation=True, max_length=1024)
        # For RealDataset Class
        # return [dataset.get_hf_dataset().map(tokenize_function, batched=True).select_columns(['input_ids', 'attention_mask']) for dataset in datasets]
        # For Dataset Class
        return [dataset.map(tokenize_function, batched=True).select_columns(['input_ids', 'attention_mask']) for dataset in datasets]
    

    def perplexity(self, data: Union[list, str], offset=0, max_length=0, apply_exp=True, verbose=True,
                return_as_list: bool = False, min_text_length: int = 5) -> float:
        """ Compute the perplexity of the model on a string.
            This function skips invalid or short samples and ensures valid PPL is calculated.
        """
        original_mode = self._lm.training
        self._lm.eval()

        if isinstance(data, str):  # always consider lists as input
            data = [data]

        nlls = []  # negative log likelihoods
        ctr = 0  # Number of valid tokens processed
        skipped_samples = 0  # Counter for skipped samples

        for txt in tqdm(data, desc="Compute PPL", disable=not verbose):
            # Skip very short or empty text
            if len(txt.strip()) < min_text_length:
                # print(f"Skipping short or empty sample: {txt}")
                skipped_samples += 1
                continue
            
            try:
                input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
                target_ids = input_ids.clone()

                if offset > 0:  # ignore everything up to the offset
                    target_ids[:, :offset] = -100

                tgt_len = (target_ids.size(1) - offset)
                if max_length > 0:  # ignore everything except offset:offset+max_length
                    target_ids[:, offset + max_length:] = -100
                    tgt_len = max_length

                with torch.no_grad():
                    outputs = self._lm(input_ids, labels=target_ids)
                loss, logits = outputs[:2]

                # Skip the sample if the loss is NaN
                if torch.isnan(loss):
                    print(f"NaN loss encountered for sample: {txt}")
                    skipped_samples += 1
                    continue

                # Accumulate valid losses
                if return_as_list:
                    nlls.append(loss.cpu().detach())
                else:
                    nlls.append(loss.cpu().detach())
                    ctr += tgt_len  # Count only valid tokens

            except Exception as e:
                print(f"Error processing sample: {txt}, Error: {e}")
                skipped_samples += 1
                continue

        self._lm.training = original_mode

        if len(nlls) == 0:
            print(f"All samples were skipped. Unable to compute PPL.")
            return float('nan')  # Return NaN if no valid samples

        # Calculate and return the final PPL based on valid samples
        if return_as_list:
            if apply_exp:
                return torch.exp(torch.stack(nlls))
            return torch.stack(nlls, 0)

        # Final perplexity calculation
        if apply_exp:
            return float(torch.exp(torch.stack(nlls).mean()).item())
        return float(torch.stack(nlls).mean().item())
    

    def get_ll(self, text: str):
        device = self.env_args.device
        original_mode = self._lm.training
        self._lm.eval()

        # NOTE: MIMIR does not use truncation here.
        input_ids = self._tokenizer.encode(text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

        # Stride and max_length were used in the MIMIR codebase.
        # We follow their practice for consistency.
        stride = self._tokenizer.model_max_length
        num_tokens = input_ids.size(1)
        total_loss = 0
        total_num_preds = 0
        # Use a sliding window when the number of input tokens exceeds
        # context window size
        for i in range(0, num_tokens, stride):
            begin_loc = i
            end_loc = min(i + stride, num_tokens)
            input_ids_i = input_ids[:, begin_loc:end_loc].to(device)
            target_ids_i = input_ids_i.clone()

            outputs = self._lm(input_ids_i, labels=target_ids_i)
            num_preds_i = target_ids_i.size(1) - 1
            total_loss += outputs.loss.item() * num_preds_i
            total_num_preds += num_preds_i

            del input_ids_i
            del target_ids_i

        self._lm.training = original_mode
        if total_num_preds == 0:
            return 10

        return total_loss / total_num_preds


    def get_probabilities(self, text: str):
        device = self.env_args.device
        original_mode = self._lm.training
        self._lm.eval()

        # NOTE: MIMIR does not use truncation here.
        input_ids = self._tokenizer.encode(text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        
        target_token_log_prob = []
        stride = self._tokenizer.model_max_length
        num_tokens = input_ids.size(1)
        for i in range(0, num_tokens, stride):
            begin_loc = i
            end_loc = min(i + stride, num_tokens)
            input_ids_i = input_ids[:, begin_loc:end_loc].to(device)
            target_ids_i = input_ids_i.clone()
            
            logits_i = self._lm(input_ids_i, labels=target_ids_i).logits
            shift_logits_i = logits_i[..., :-1, :].contiguous()
            log_probabilities_i = torch.nn.functional.log_softmax(shift_logits_i, dim=-1)
            shift_labels_i = target_ids_i[..., 1:].contiguous()
            labels_processed_i = shift_labels_i[0]

            del input_ids_i
            del target_ids_i
            
            for j, token_id in enumerate(labels_processed_i):
                target_token_log_prob.append(
                    log_probabilities_i[0, j, token_id].item()
                )
        
        return target_token_log_prob


    def _fine_tune_dp(self,
                      train_dataset: RealDataset,
                      eval_dataset: RealDataset,
                      train_args: TrainerArgs,
                      privacy_args: PrivacyArgs):

        with train_args.main_process_first(desc="Tokenizing datasets"):
            hf_train_dataset, hf_eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])

        self._lm = self._lm.to(self.env_args.device)
        self._lm.train()

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self._tokenizer)

        # transfer privacy args
        dpt_privacy_args = dp_transformers.PrivacyArguments(noise_multiplier=privacy_args.noise_multiplier,
                                                            per_sample_max_grad_norm=privacy_args.max_grad_norm_dp)

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=train_args,
            model=self._lm,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_eval_dataset,
            data_collator=data_collator,
            privacy_args=dpt_privacy_args,
            tokenizer=self._tokenizer
        )

        trainer.use_cuda_amp = False

        try:
            trainer.train()
        finally:
            trainer.save_model()
            self._lm.save_pretrained(trainer.args.output_dir)
            self._tokenizer.save_pretrained(trainer.args.output_dir)
        self._lm.eval()

    def fine_tune(self,
                  train_dataset,
                  eval_dataset,
                  train_args: TrainerArgs,
                  privacy_args: PrivacyArgs):
        """ Fine-Tune the LM with/without DP
        """
        if privacy_args.target_epsilon > 0:
            return self._fine_tune_dp(train_dataset, eval_dataset, train_args, privacy_args)
        return self._fine_tune(train_dataset, eval_dataset, train_args)

    def _fine_tune(self,
                   train_dataset,
                   eval_dataset,
                   train_args: TrainerArgs,
                   extra_callbacks: List[TrainerCallback] = None):
        """ Fine-Tune the model and save checkpoints to output directory
        """
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
                                                       num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])
        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"
        trainer = Trainer(model=self._lm,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          callbacks=extra_callbacks)

        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        trainer.save_model()
        self._lm.eval()


    def _unlearn_gradient_ascent(self,
                                unlearn_dataset,
                                eval_dataset,
                                train_args: TrainerArgs,
                                extra_callbacks: List[TrainerCallback] = None):
        """ Unlearn certain texts using gradient ascent and save checkpoints to output directory """
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Unlearn Eval PPL",
                                                    num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Unlearn and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        unlearn_dataset, eval_dataset = self.tokenize_datasets([unlearn_dataset, eval_dataset])
        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"  
        unlearner = GradientAscentTrainer(model=self._lm,
                                        args=train_args,
                                        train_dataset=unlearn_dataset,
                                        eval_dataset=eval_dataset,
                                        data_collator=data_collator,
                                        callbacks=extra_callbacks)

        unlearner.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        unlearner.save_model()
        self._lm.eval()

    def _unlearn_retrain(self,
                        retain_dataset,
                        eval_dataset,
                        train_args: TrainerArgs,
                        extra_callbacks: List[TrainerCallback] = None):
        """ Retrain the model on the retain dataset and save checkpoints to output directory """
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Unlearn Eval PPL",
                                                    num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Retain and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        retain_dataset, eval_dataset = self.tokenize_datasets([retain_dataset, eval_dataset])
        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"
        unlearner = Trainer(model=self._lm,
                            args=train_args,
                            train_dataset=retain_dataset,
                            eval_dataset=eval_dataset,
                            data_collator=data_collator,
                            callbacks=extra_callbacks)

        unlearner.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        unlearner.save_model()
        self._lm.eval()

    def _unlearn_random_label(self,
                            unlearn_dataset,
                            eval_dataset,
                            train_args: TrainerArgs,
                            extra_callbacks: List[TrainerCallback] = None):
        """ Unlearn certain texts using random label method and save checkpoints to output directory """
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Unlearn Eval PPL",
                                                    num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Unlearn and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        unlearn_dataset, eval_dataset = self.tokenize_datasets([unlearn_dataset, eval_dataset])
        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"
        train_args.save_strategy = "epoch"  
        train_args.save_steps = None
        unlearner = RandomLabelTrainer(model=self._lm,
                                    args=train_args,
                                    train_dataset=unlearn_dataset,
                                    eval_dataset=eval_dataset,
                                    data_collator=data_collator,
                                    callbacks=extra_callbacks)

        unlearner.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        unlearner.save_model()
        self._lm.eval()


    def _unlearn_ascent_plus_descent(self, 
                                    unlearn_dataset, 
                                    retain_dataset, 
                                    eval_dataset, 
                                    train_args: TrainerArgs, 
                                    extra_callbacks: List[TrainerCallback] = None):
        """Unlearn and retain certain texts using gradient ascent and descent, then save checkpoints."""
        
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Unlearn Eval PPL",
                                                num_steps=train_args.callback_after_n_steps)]
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Unlearn, Retain, and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        unlearn_dataset, eval_dataset, retain_dataset = self.tokenize_datasets([unlearn_dataset, eval_dataset, retain_dataset])

        # Prepare retain set for each epoch by combining them with unlearn set
        combined_datasets = []
        retain_size_per_epoch = len(unlearn_dataset)

        for epoch in range(train_args.num_train_epochs):
            # Shuffle retain set for each epoch and select same size as unlearn set
            retain_samples = retain_dataset.shuffle().select(range(retain_size_per_epoch))

            # Assign -1 to unlearn set and 1 to retain set
            unlearn_epoch_dataset = unlearn_dataset.add_column("factor", [-1] * len(unlearn_dataset))
            retain_epoch_dataset = retain_samples.add_column("factor", [1] * len(retain_samples))

            # Combine the unlearn set and retain set
            epoch_combined_set = concatenate_datasets([unlearn_epoch_dataset, retain_epoch_dataset]).shuffle()

            # Add to the list of combined datasets
            combined_datasets.append(epoch_combined_set)

        # Concatenate all epochs' datasets into one large combined dataset
        combined_dataset  = concatenate_datasets(combined_datasets)

        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"
        train_args.dataloader_shuffle = False  # Disable shuffling during training
        train_args.num_train_epochs = 1

        # Initialize the AscentPlusDescentTrainer with the prepared combined dataset
        trainer = AscentPlusDescentTrainer(model=self._lm,
                                        args=train_args,
                                        train_dataset=combined_dataset, 
                                        eval_dataset=eval_dataset,
                                        data_collator=data_collator,
                                        callbacks=extra_callbacks)

        # Start training
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

        # Save the model after training
        trainer.save_model()

        # Set the model back to evaluation mode
        self._lm.eval()

        
    def _unlearn_langevin(self,
                        retain_dataset,
                        eval_dataset,
                        train_args: TrainerArgs,
                        privacy_args: PrivacyArgs,
                        noise_scale: float,
                        extra_callbacks: List[TrainerCallback] = None):
        with train_args.main_process_first(desc="Tokenizing datasets"):
            hf_train_dataset, hf_eval_dataset = self.tokenize_datasets([retain_dataset, eval_dataset])

        self._lm = self._lm.to(self.env_args.device)
        self._lm.train()

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self._tokenizer)

        # transfer privacy args
        dpt_privacy_args = dp_transformers.PrivacyArguments(noise_multiplier=privacy_args.noise_multiplier,
                                                            per_sample_max_grad_norm=privacy_args.max_grad_norm_dp)

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=train_args,
            model=self._lm,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_eval_dataset,
            data_collator=data_collator,
            privacy_args=dpt_privacy_args,
            tokenizer=self._tokenizer
        )

        trainer.use_cuda_amp = False

        try:
            trainer.train()
        finally:
            trainer.save_model()
            self._lm.save_pretrained(trainer.args.output_dir)
            self._tokenizer.save_pretrained(trainer.args.output_dir)
        self._lm.eval()


    def _unlearn_EUk(self,
                    retain_dataset,
                    eval_dataset,
                    train_args: TrainerArgs,
                    k_layers: int,
                    extra_callbacks: List[TrainerCallback] = None):
        """ Unlearn certain texts using EUk method and save checkpoints to output directory """
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Unlearn Eval PPL",
                                                    num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Retain and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        retain_dataset, eval_dataset = self.tokenize_datasets([retain_dataset, eval_dataset])
        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"

        # Freeze all layers except the last k layers
        total_params = 0
        unfrozen_params = 0

        # Freeze all layers except the last k layers
        for name, param in self._lm.named_parameters():
            param.requires_grad = False
            total_params += param.numel()

        total_layers = len(list(self._lm.transformer.h))
        for i in range(total_layers - k_layers, total_layers):
            for name, param in self._lm.transformer.h[i].named_parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

        # Re-initialize the last k layers
        for i in range(total_layers - k_layers, total_layers):
            self._lm.transformer.h[i].apply(self._lm._init_weights)

        unfrozen_ratio = (unfrozen_params / total_params) * 100

        # Only the last k layers will be trained
        unlearner = Trainer(model=self._lm,
                            args=train_args,
                            train_dataset=retain_dataset,
                            eval_dataset=eval_dataset,
                            data_collator=data_collator,
                            callbacks=extra_callbacks)

        unlearner.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        unlearner.save_model()
        self._lm.eval()

    def _unlearn_CFk(self,
                    retain_dataset,
                    eval_dataset,
                    train_args: TrainerArgs,
                    k_layers: int,
                    extra_callbacks: List[TrainerCallback] = None):
        """ Unlearn certain texts using CFk method and save checkpoints to output directory """
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Unlearn Eval PPL",
                                                    num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Retain and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        retain_dataset, eval_dataset = self.tokenize_datasets([retain_dataset, eval_dataset])
        print("Done Tokenizing!")

        train_args.evaluation_strategy = "no"

        # Freeze all layers except the last k layers
        for name, param in self._lm.named_parameters():
            param.requires_grad = False

        total_layers = len(list(self._lm.transformer.h))
        for i in range(total_layers - k_layers, total_layers):
            for name, param in self._lm.transformer.h[i].named_parameters():
                param.requires_grad = True

        # Only the last k layers will be trained on retain dataset without re-initialization
        unlearner = Trainer(model=self._lm,
                            args=train_args,
                            train_dataset=retain_dataset,
                            eval_dataset=eval_dataset,
                            data_collator=data_collator,
                            callbacks=extra_callbacks)

        unlearner.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        unlearner.save_model()
        self._lm.eval()
        
    def _unlearn_scrub(self,
                    unlearn_dataset,
                    retain_dataset,
                    eval_dataset,
                    train_args: TrainerArgs,
                    alpha: float,
                    beta: float,
                    gamma: float,
                    extra_callbacks: List[TrainerCallback] = None):
        """Unlearn certain texts using the scrub method and save checkpoints to output directory"""
        
        if extra_callbacks is None:
            extra_callbacks = []

        extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
                                                num_steps=train_args.callback_after_n_steps)]
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Unlearn Eval PPL",
                                                    num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Unlearn, Retain, and Eval Datasets...")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        unlearn_dataset, eval_dataset, retain_dataset = self.tokenize_datasets([unlearn_dataset, eval_dataset, retain_dataset])

        # Prepare the combined dataset for multiple epochs
        combined_datasets = []

        for epoch in range(train_args.num_train_epochs):
            # Shuffle and select the retain set for the current epoch (same size as unlearn set)
            retain_epoch_dataset = retain_dataset.shuffle().select(range(len(unlearn_dataset)))
            
            # Assign factor -1 to unlearn and 1 to retain dataset
            unlearn_epoch_dataset = unlearn_dataset.add_column("factor", [-1] * len(unlearn_dataset))
            retain_epoch_dataset = retain_epoch_dataset.add_column("factor", [1] * len(retain_epoch_dataset))
            
            # Shuffle within the current epoch and combine unlearn and retain sets
            epoch_combined_dataset = concatenate_datasets([unlearn_epoch_dataset, retain_epoch_dataset]).shuffle()
            
            combined_datasets.append(epoch_combined_dataset)

        # Concatenate all the epoch datasets into a single dataset for training
        combined_dataset = concatenate_datasets(combined_datasets)

        print("Done preparing combined datasets for all epochs!")

        initial_model = copy.deepcopy(self._lm).eval()  # Copy the initial model for KL divergence

        # Set the necessary arguments for sequential training
        train_args.evaluation_strategy = "no"
        train_args.dataloader_shuffle = False  # Disable shuffling during training
        train_args.save_strategy = "steps"
        train_args.save_steps = 3
        train_args.num_train_epochs = 1

        # Initialize the SCRUBTrainer with the combined dataset
        scrub_trainer = SCRUBTrainer(model=self._lm,
                                    initial_model=initial_model, 
                                    args=train_args,
                                    train_dataset=combined_dataset,  # Pass the prepared combined dataset
                                    eval_dataset=eval_dataset,  # Evaluation dataset
                                    data_collator=data_collator,
                                    callbacks=extra_callbacks,
                                    alpha=alpha,
                                    beta=beta,
                                    gamma=gamma)

        # Start training
        scrub_trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

        # Save the model after training
        scrub_trainer.save_model()

        # Set the model back to evaluation mode
        self._lm.eval()


    def unlearn(self,
                unlearn_dataset,
                retain_dataset,
                eval_dataset,
                train_args: TrainerArgs,
                privacy_args: PrivacyArgs,
                unlearn_method: str = "EUk",
                noise_scale: float = 1.0,
                k_layers: int = 1,
                alpha: float = 0.5,
                beta: float = 1,
                gamma: float = 0.01,
                extra_callbacks: List[TrainerCallback] = None):
        """ Unlearn certain texts from the model using specified method and save checkpoints to output directory """
        if unlearn_method == "gradient_ascent":
            return self._unlearn_gradient_ascent(unlearn_dataset, eval_dataset, train_args, extra_callbacks)
        elif unlearn_method == "retrain":
            return self._unlearn_retrain(retain_dataset, eval_dataset, train_args, extra_callbacks)
        elif unlearn_method == "random_label":
            return self._unlearn_random_label(unlearn_dataset, eval_dataset, train_args, extra_callbacks)
        elif unlearn_method == "ascent_plus_descent":
            return self._unlearn_ascent_plus_descent(unlearn_dataset, retain_dataset, eval_dataset, train_args, extra_callbacks)
        elif unlearn_method == "langevin":
            return self._unlearn_langevin(retain_dataset, eval_dataset, train_args, privacy_args, noise_scale, extra_callbacks)
        elif unlearn_method == "EUk":
            return self._unlearn_EUk(retain_dataset, eval_dataset, train_args, k_layers, extra_callbacks)
        elif unlearn_method == "CFk":
            return self._unlearn_CFk(retain_dataset, eval_dataset, train_args, k_layers, extra_callbacks)
        elif unlearn_method == "scrub":
            return self._unlearn_scrub(unlearn_dataset, retain_dataset, eval_dataset, train_args, alpha, beta, gamma, extra_callbacks)
        else:
            raise NotImplementedError(f"Unlearn method '{unlearn_method}' is not implemented")
