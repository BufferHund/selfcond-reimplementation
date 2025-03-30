#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptDataset(Dataset):
    """Dataset class for concept data"""
    def __init__(
        self,
        json_file: Path,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        num_samples: int = 1000,
        random_seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load concept data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {json_file}, error: {str(e)}")
            raise
            
        # Get positive and negative examples
        sentences = data.get('sentences', {})
        self.positive_examples = sentences.get('positive', [])
        self.negative_examples = sentences.get('negative', [])
        
        if not self.positive_examples and not self.negative_examples:
            logger.warning(f"No examples found in file {json_file}")
            raise ValueError("Empty dataset")
            
        logger.info(f"Loaded {len(self.positive_examples)} positive and {len(self.negative_examples)} negative examples")
        
        # Random sampling
        np.random.seed(random_seed)
        self.positive_examples = np.random.choice(
            self.positive_examples, 
            min(len(self.positive_examples), num_samples),
            replace=False
        )
        self.negative_examples = np.random.choice(
            self.negative_examples,
            min(len(self.negative_examples), num_samples),
            replace=False
        )
        
        self.examples = list(self.positive_examples) + list(self.negative_examples)
        self.labels = [1] * len(self.positive_examples) + [0] * len(self.negative_examples)
        
        logger.info(f"Sampled {len(self.examples)} examples")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            text = self.examples[idx]
            label = self.labels[idx]
            
            # Encode text
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': label
            }
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            raise

class ModelResponseCollector:
    """Class for collecting model responses"""
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir: Optional[Path] = None
    ):
        self.device = device
        logger.info(f"Loading model {model_name} to {device}")
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                output_hidden_states=True
            ).to(device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
    def get_layer_responses(self, batch):
        """Get responses from each layer"""
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                
            # Collect hidden states
            hidden_states = outputs.hidden_states
            responses = {}
            
            # GPT2-medium has 24 layers
            for layer_idx, hidden_state in enumerate(hidden_states):
                if layer_idx >= 24:  # Skip layer 24
                    continue
                # Max pooling for each token
                pooled = torch.max(hidden_state, dim=1)[0]
                responses[f'layer_{layer_idx}'] = pooled.cpu().numpy()
                
            return responses
        except Exception as e:
            logger.error(f"Error getting model responses: {str(e)}")
            raise

def process_concept(
    concept_name: str,
    concept_group: str,
    data_path: Path,
    save_path: Path,
    model_collector: ModelResponseCollector,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    num_samples: int = 1000,
) -> None:
    """Process responses for a single concept"""
    try:
        json_file = data_path / concept_group / f"{concept_name}.json"
        if not json_file.exists():
            logger.warning(f"Skipping {json_file}, file not found")
            return
            
        save_dir = save_path / concept_group / concept_name
        if (save_dir / "responses").exists():
            logger.info(f"Skipping {concept_group}/{concept_name}, responses already computed")
            return
            
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset and dataloader
        logger.info(f"Creating dataset: {json_file}")
        dataset = ConceptDataset(
            json_file=json_file,
            tokenizer=tokenizer,
            max_length=max_length,
            num_samples=num_samples
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Collect responses
        all_responses = {}
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {concept_group}/{concept_name}")):
            try:
                responses = model_collector.get_layer_responses(batch)
                
                # Merge batch responses
                for layer_name, layer_responses in responses.items():
                    if layer_name not in all_responses:
                        all_responses[layer_name] = []
                    all_responses[layer_name].append(layer_responses)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue
        
        # Save responses
        responses_dir = save_dir / "responses"
        responses_dir.mkdir(exist_ok=True)
        
        for layer_name, responses in all_responses.items():
            try:
                responses = np.concatenate(responses, axis=0)
                save_file = responses_dir / f"{layer_name}.npy"
                np.save(save_file, responses)
                logger.info(f"Saved responses to {save_file}")
            except Exception as e:
                logger.error(f"Error saving layer {layer_name} responses: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error processing concept {concept_group}/{concept_name}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Compute model responses for concepts")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name")
    parser.add_argument("--data-path", required=True, type=Path, help="Path to concept data")
    parser.add_argument("--save-path", required=True, type=Path, help="Path to save responses")
    parser.add_argument("--concepts", help="List of concepts to process (comma-separated)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--device", help="Device to run on")
    parser.add_argument("--cache-dir", type=Path, help="Model cache directory")
    
    args = parser.parse_args()
    
    try:
        # Set device
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize model and tokenizer
        logger.info("Initializing model and tokenizer...")
        model_collector = ModelResponseCollector(
            args.model_name,
            device=device,
            cache_dir=args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Get concept list
        if args.concepts:
            concepts = args.concepts.split(',')
            concept_df = pd.DataFrame([
                {'concept': c.split('/')[1], 'group': c.split('/')[0]}
                for c in concepts
            ])
        else:
            concept_df = pd.read_csv(args.data_path / "concept_list.csv")
        
        logger.info(f"Found {len(concept_df)} concepts to process")
        
        # Process each concept
        for _, row in tqdm(concept_df.iterrows(), total=len(concept_df)):
            concept, group = row['concept'], row['group']
            if concept in ['positive', 'negative'] and group == 'keyword':
                continue
                
            logger.info(f"Processing concept {group}/{concept}")
            process_concept(
                concept_name=concept,
                concept_group=group,
                data_path=args.data_path,
                save_path=args.save_path,
                model_collector=model_collector,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_samples=args.num_samples
            )
            
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
