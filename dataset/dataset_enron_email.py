import os
import re
import torch
import spacy
import random
import numpy as np
from collections import Counter
from datasets import load_dataset, Dataset

# Configuration
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def extract_sentences_near_email(text):
    """Extract sentences near email addresses."""
    sentences = text.split('. ')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b')
    extracted_sentences = []
    detected_email_domain = None

    # Try to find sentences with exactly one email address
    email_matches = re.findall(email_pattern, text)
    
    if len(email_matches) == 1:
        # Email domain detected
        detected_email_domain = email_matches[0]
        for i, sentence in enumerate(sentences):
            if re.search(email_pattern, sentence):
                # Get three coherent sentences (current, previous, and next)
                extracted = sentences[max(0, i-1):min(len(sentences), i+2)]
                extracted_sentences.extend(extracted)
                break  # Stop after finding the first email
    
    # If no email or more than one email is found, randomly select three consecutive sentences
    if not extracted_sentences:
        if len(sentences) >= 3:
            random_start = random.randint(0, len(sentences) - 3)
            extracted_sentences = sentences[random_start:random_start + 3]
        else:
            extracted_sentences = sentences  # If less than 3 sentences, return all sentences

    return '. '.join(extracted_sentences), detected_email_domain

def process_dataset(dataset):
    """Clean the email text and extract sentences near email addresses for each dataset."""
    cleaned_texts = []
    email_domains = []
    
    for sample in dataset:
        coherent_text, email_domain = extract_sentences_near_email(sample['text'])
        cleaned_texts.append(coherent_text)
        email_domains.append(email_domain if email_domain else "No Email")  # Add "No Email" if no domain found

    return Dataset.from_dict({"text": cleaned_texts, "email_domain": email_domains})

def main():
    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the public Enron dataset
    dataset_dict = load_dataset("snoop2head/enron_aeslc_emails")
    print(f"Available dataset splits: {dataset_dict.keys()}")

    # Sample 10,000 training samples and 10,000 eval samples
    train_dataset = dataset_dict['train'].select(range(10000))
    eval_dataset = dataset_dict['train'].select(range(10000, 20000))
    remain_dataset = dataset_dict['train'].select(range(20000, len(dataset_dict['train'])))

    # Process train, eval, and remain datasets
    print("Processing train, eval, and remain datasets...")
    processed_train_dataset = process_dataset(train_dataset)
    processed_eval_dataset = process_dataset(eval_dataset)
    processed_remain_dataset = process_dataset(remain_dataset)

    # Collect all email domains from the remain dataset
    domain_counter = Counter()
    for sample in processed_remain_dataset['email_domain']:
        if sample != "No Email":  # Only count actual email domains
            domain_counter.update([sample])

    # Get least frequent email domains for the minority set
    r = 100  # Define size of the minority set
    least_frequent_domains = [domain for domain, freq in domain_counter.most_common()[:-r-1:-1]]
    minority_set = [sample for sample in processed_remain_dataset if sample['email_domain'] in least_frequent_domains]

    # Define relative save directory paths under source_dataset/enron_email
    current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_directory = os.path.join(current_path, "source_dataset", "enron_email")
    os.makedirs(save_directory, exist_ok=True)

    # Save the processed datasets
    print("Saving datasets...")
    processed_train_dataset.save_to_disk(os.path.join(save_directory, "enron_email_train"))
    processed_eval_dataset.save_to_disk(os.path.join(save_directory, "enron_email_test"))
    minority_set_dataset = Dataset.from_dict({
        "text": [sample['text'] for sample in minority_set],
        "email_domain": [sample['email_domain'] for sample in minority_set]
    })
    minority_set_dataset.save_to_disk(os.path.join(save_directory, "enron_email_minority"))

    print("Datasets saved successfully.")

    # Generate perturb set and no-perturb set
    print("Generating perturb set and no-perturb set...")

    # Filter samples from processed_remain_dataset that have email addresses
    samples_with_email = [sample for sample in processed_remain_dataset if sample['email_domain'] != "No Email"]

    # Shuffle and select r samples for no-perturb set
    random.shuffle(samples_with_email)
    no_perturb_samples = samples_with_email[:r]

    # Create perturb set by replacing email domains in the text with 'minister.com'
    perturb_samples = []
    for sample in no_perturb_samples:
        perturbed_text = re.sub(
            r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
            r'\1@minister.com',
            sample['text']
        )
        perturb_samples.append({'text': perturbed_text, 'email_domain': 'minister.com'})

    # Create datasets for no-perturb and perturb sets
    no_perturb_dataset = Dataset.from_dict({
        'text': [sample['text'] for sample in no_perturb_samples],
        'email_domain': [sample['email_domain'] for sample in no_perturb_samples]
    })

    perturb_dataset = Dataset.from_dict({
        'text': [sample['text'] for sample in perturb_samples],
        'email_domain': [sample['email_domain'] for sample in perturb_samples]
    })

    # Save the datasets to disk
    no_perturb_dataset.save_to_disk(os.path.join(save_directory, "enron_email_no_perturb"))
    perturb_dataset.save_to_disk(os.path.join(save_directory, "enron_email_perturb"))

    print("Perturb set and no-perturb set saved successfully.")

if __name__ == "__main__":
    main()
