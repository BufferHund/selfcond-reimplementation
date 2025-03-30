#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# Original file from:
#
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Conditional text generation with the auto-regressive models of the HuggingFace Transformers repository.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    prompt: str = "EOS"
    length: int = 20
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.9
    eos: bool = True
    verbose: bool = False

@dataclass
class InterventionConfig:
    """Configuration for neuron intervention"""
    forcing_values: List[str]
    num_units: List[int]
    top_n: List[int]
    per_layer: bool = False
    only_last_token: bool = False
    metric: str = "ap"

class TextGenerator:
    """Text generator with neuron intervention"""
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        # Force CPU usage
        self.device = 'cpu'
        self.n_gpu = 0
        logger.info(f"Using device: {self.device} ({self.n_gpu})")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelWithLMHead.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()
        
        # Store original hidden states and hooks
        self.original_hidden_states = {}
        self.hooks = []
        self._setup_hooks()
        
    def _setup_hooks(self):
        """Setup model hooks to capture hidden states"""
        def hook_fn(name):
            def hook(module, input, output):
                self.original_hidden_states[name] = output[0].detach()
            return hook
            
        for name, module in self.model.named_modules():
            if 'h.' in name:  # Only set hooks for transformer layers
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append((name, hook))
                
    def _generate_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Tuple[torch.Tensor, float]:
        """Generate a single token"""
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                # Apply nucleus sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Compute probability distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample on CPU
                probs_cpu = probs.cpu()
                next_token = torch.multinomial(probs_cpu, num_samples=1)
                return next_token.to(self.device), float(probs_cpu[0, next_token[0, 0]])
        except Exception as e:
            logger.error(f"Error generating token: {str(e)}")
            raise
            
    def generate_text(
        self,
        config: GenerationConfig,
        expertise_df: pd.DataFrame,
        intervention_config: InterventionConfig,
        seeds: List[int]
    ) -> pd.DataFrame:
        """Generate text with neuron intervention"""
        # Save expertise_df for hook functions
        self.expertise_df = expertise_df
        
        results = []
        
        # Extract concept name from file path
        expertise_path = Path(expertise_df["expertise_path"].iloc[0])
        concept = expertise_path.parent.parent.name
        
        # Get layer names
        layer_names = (
            list(expertise_df.sort_values("layer").layer.unique())
            if intervention_config.per_layer
            else [None]
        )
        
        for forcing_value in intervention_config.forcing_values:
            for top_n in intervention_config.top_n:
                for force_layer in layer_names:
                    for num_units in intervention_config.num_units:
                        logger.info(
                            f"Generating [force={forcing_value} units={num_units}/{len(expertise_df)} "
                            f"({100 * num_units / len(expertise_df):.3f}%) "
                            f"top_n={top_n} layers={force_layer}]"
                        )
                        
                        # Setup neuron intervention
                        mean_metric = 0
                        if num_units > 0:
                            mean_metric = self._setup_intervention(
                                expertise_df,
                                forcing_value,
                                intervention_config.metric,
                                num_units,
                                top_n,
                                force_layer,
                                intervention_config.only_last_token
                            )
                        
                        # Generate text for each seed
                        for seed in seeds:
                            # Set random seeds
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            if self.device != 'cpu':
                                torch.cuda.manual_seed(seed)
                            
                            # Add random noise to intervention values
                            if num_units > 0:
                                for hook_name, hook in self.hooks:
                                    if hook_name.startswith("intervention_"):
                                        original_value = hook.intervention_value
                                        noise = np.random.normal(0, 0.01)
                                        hook.intervention_value = original_value + noise
                            
                            if config.verbose:
                                logger.info(f"\n{concept} s={seed} f={num_units}:")
                                
                            # Generate text
                            text, perplexity = self._generate_sequence(config)
                            
                            # Add result
                            result = {
                                "forcing_value": forcing_value,
                                "num_units": num_units,
                                "top_n": top_n,
                                "seed": seed,
                                "sentence": text,
                                "mean_metric": mean_metric,
                                "forced_layer": force_layer,
                                "perplexity": perplexity,
                                "context": config.prompt,
                                "concept": concept
                            }
                            results.append(result)
                            
                        # Restore original hidden states
                        if num_units > 0:
                            self._restore_hidden_states()
                            
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
        
    def _generate_sequence(self, config: GenerationConfig) -> Tuple[str, float]:
        """Generate complete sequence"""
        # Encode input
        input_ids = self.tokenizer.encode(config.prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_tokens = []
        log_probs = []
        
        # Record original prompt tokens
        original_tokens = input_ids[0].tolist()
        
        for _ in range(config.length):
            next_token, prob = self._generate_token(
                input_ids,
                attention_mask,
                config.temperature,
                config.top_k,
                config.top_p
            )
            
            # Check if generated token is valid
            if len(generated_tokens) == 0 and next_token.item() in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                # Skip if first token is EOS or PAD
                continue
                
            generated_tokens.append(next_token.item())
            log_probs.append(float(np.log(prob)))
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            if config.eos and next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # Decode generated text
        text = self.tokenizer.decode(generated_tokens)
        
        # Clean text
        text = text.strip()
        if text.startswith(','):
            text = text[1:].strip()
        if text.startswith('.'):
            text = text[1:].strip()
            
        # Compute perplexity
        perplexity = float(np.exp(-np.mean(log_probs)))
        
        return text, perplexity
        
    def _setup_intervention(
        self,
        expertise_df: pd.DataFrame,
        forcing_value: str,
        metric: str,
        num_units: int,
        top_n: int,
        force_layer: Optional[str],
        only_last_token: bool
    ) -> float:
        """Setup neuron intervention"""
        try:
            # Select top-k neurons based on metric
            if force_layer:
                layer_df = expertise_df[expertise_df["layer"] == force_layer]
            else:
                layer_df = expertise_df
                
            # Select top-k neurons
            top_neurons = layer_df.nlargest(num_units, metric)
            
            # Compute mean metric
            mean_metric = float(top_neurons[metric].mean())
            
            # Get model configuration
            num_layers = len(self.model.transformer.h)
            hidden_size = self.model.config.hidden_size
            
            logger.info(f"Model config: layers={num_layers}, hidden_size={hidden_size}")
            
            # Set intervention values
            for _, neuron in top_neurons.iterrows():
                try:
                    # Get layer and neuron indices
                    layer_name = neuron["layer"]
                    layer_idx = int(layer_name.split('_')[1])
                    neuron_idx = int(neuron["neuron_idx"])
                    
                    # Ensure indices are valid
                    if layer_idx >= num_layers:
                        logger.warning(f"Skipping layer {layer_idx}, exceeds layer count {num_layers}")
                        continue
                        
                    if neuron_idx >= hidden_size:
                        logger.warning(f"Skipping neuron {neuron_idx}, exceeds hidden size {hidden_size}")
                        continue
                    
                    # Get intervention value
                    if forcing_value == "on_p50":
                        value = float(neuron["on_p50"])
                    else:  # off_p50
                        value = float(neuron["off_p50"])
                    
                    # Log debug info
                    logger.info(f"Setting intervention: layer={layer_idx}, neuron={neuron_idx}, value={value}")
                    
                    # Apply intervention
                    if only_last_token:
                        self._intervene_on_last_token(layer_idx, neuron_idx, value)
                    else:
                        self._intervene_on_all_tokens(layer_idx, neuron_idx, value)
                except Exception as e:
                    logger.warning(f"Error processing neuron {neuron_idx}: {str(e)}")
                    continue
                    
            return mean_metric
        except Exception as e:
            logger.error(f"Error setting up intervention: {str(e)}")
            raise
        
    def _intervene_on_last_token(
        self,
        layer_idx: int,
        neuron_idx: int,
        value: float
    ):
        """Intervene on last token"""
        def hook(module, input, output):
            try:
                hidden_states = output[0]
                # Ensure index is valid
                if neuron_idx < hidden_states.size(-1):
                    # Use CPU to avoid CUDA errors
                    hidden_states_cpu = hidden_states.cpu()
                    hidden_states_cpu[:, -1, neuron_idx] = value
                    hidden_states.copy_(hidden_states_cpu.to(hidden_states.device))
                    logger.debug(f"Set layer {layer_idx} neuron {neuron_idx} to {value}")
                else:
                    logger.warning(f"Neuron index {neuron_idx} exceeds hidden size {hidden_states.size(-1)}")
            except Exception as e:
                logger.warning(f"Error in hook for layer {layer_idx} neuron {neuron_idx}: {str(e)}")
            return output
            
        hook = self.model.transformer.h[layer_idx].register_forward_hook(hook)
        # Store intervention value
        hook.intervention_value = value
        self.hooks.append((f"intervention_{layer_idx}_{neuron_idx}", hook))
        
    def _intervene_on_all_tokens(
        self,
        layer_idx: int,
        neuron_idx: int,
        value: float
    ):
        """Intervene on all tokens"""
        def hook(module, input, output):
            try:
                hidden_states = output[0]
                # Ensure index is valid
                if neuron_idx < hidden_states.size(-1):
                    # Use CPU to avoid CUDA errors
                    hidden_states_cpu = hidden_states.cpu()
                    hidden_states_cpu[:, :, neuron_idx] = value
                    hidden_states.copy_(hidden_states_cpu.to(hidden_states.device))
                    logger.debug(f"Set layer {layer_idx} neuron {neuron_idx} to {value}")
                else:
                    logger.warning(f"Neuron index {neuron_idx} exceeds hidden size {hidden_states.size(-1)}")
            except Exception as e:
                logger.warning(f"Error in hook for layer {layer_idx} neuron {neuron_idx}: {str(e)}")
            return output
            
        hook = self.model.transformer.h[layer_idx].register_forward_hook(hook)
        # Store intervention value
        hook.intervention_value = value
        self.hooks.append((f"intervention_{layer_idx}_{neuron_idx}", hook))
        
    def _restore_hidden_states(self):
        """Restore original hidden states"""
        # Remove all intervention hooks
        for name, hook in self.hooks:
            if name.startswith("intervention_"):
                hook.remove()
        self.hooks = [(name, hook) for name, hook in self.hooks if not name.startswith("intervention_")]

def main():
    parser = argparse.ArgumentParser(description="Generate text with neuron intervention")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--expertise", required=True, type=Path, help="Expertise analysis CSV file")
    parser.add_argument("--cache-dir", type=Path, help="Model cache directory")
    parser.add_argument("--prompt", default="EOS", help="Generation prompt")
    parser.add_argument("--length", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--device", help="Device to run on")
    parser.add_argument("--seed", type=int, nargs="+", default=[1], help="Random seeds")
    parser.add_argument("--metric", default="ap", help="Metric for selecting experts")
    parser.add_argument("--forcing", nargs="+", default=["on_p50"], help="Intervention values (on_p50 or off_p50)")
    parser.add_argument("--num-units", type=int, nargs="+", default=[1], help="Number of neurons to intervene")
    parser.add_argument("--top-n", type=int, nargs="+", default=[1], help="Which top neurons to use")
    parser.add_argument("--per-layer", action="store_true", help="Use specified number of neurons per layer")
    parser.add_argument("--eos", action="store_true", help="Stop at EOS")
    parser.add_argument("--verbose", action="store_true", help="Show detailed info")
    parser.add_argument("--only-last-token", action="store_true", help="Only intervene on last token")
    parser.add_argument("--results-file", type=Path, help="Results save file")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    try:
        # Create generator
        generator = TextGenerator(
            model_name=args.model_name,
            device=args.device,
            cache_dir=args.cache_dir
        )
        
        # Load expertise analysis results
        expertise_df = pd.read_csv(args.expertise)
        expertise_df["expertise_path"] = str(args.expertise)
        
        # Validate forcing values
        valid_forcing_values = ["on_p50", "off_p50"]
        for value in args.forcing:
            if value not in valid_forcing_values:
                raise ValueError(f"forcing value must be one of {valid_forcing_values}, not {value}")
        
        # Create configs
        gen_config = GenerationConfig(
            prompt=args.prompt,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos=args.eos,
            verbose=args.verbose
        )
        
        intervention_config = InterventionConfig(
            forcing_values=args.forcing,
            num_units=args.num_units,
            top_n=args.top_n,
            per_layer=args.per_layer,
            only_last_token=args.only_last_token,
            metric=args.metric
        )
        
        # Generate text
        results_df = generator.generate_text(
            config=gen_config,
            expertise_df=expertise_df,
            intervention_config=intervention_config,
            seeds=args.seed
        )
        
        # Save results
        if not args.no_save:
            if args.results_file:
                results_file = args.results_file
            else:
                results_file = args.expertise.parent / f'forced_sentences_{args.expertise.parent.parent.name}_{args.prompt.replace("_", "")}.csv'
                
            if results_file.exists():
                # Read existing results
                previous_df = pd.read_csv(results_file)
                
                # Check for duplicates
                new_results = []
                for _, row in results_df.iterrows():
                    # Check if result is duplicate
                    is_duplicate = False
                    for _, prev_row in previous_df.iterrows():
                        if (row['sentence'] == prev_row['sentence'] and 
                            row['forcing_value'] == prev_row['forcing_value'] and
                            row['num_units'] == prev_row['num_units'] and
                            row['top_n'] == prev_row['top_n'] and
                            row['seed'] == prev_row['seed'] and
                            row['forced_layer'] == prev_row['forced_layer']):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        new_results.append(row)
                
                # Add new results
                if new_results:
                    new_df = pd.DataFrame(new_results)
                    results_df = pd.concat([previous_df, new_df], ignore_index=True)
                    # Remove duplicate index columns
                    results_df = results_df.loc[:, ~results_df.columns.str.contains('^Unnamed')]
                    # Save results
                    results_df.to_csv(results_file, index=False)
                    logger.info(f"Added {len(new_results)} new results to {results_file}")
                else:
                    logger.info("No new results to add")
            else:
                # Save new results without index
                results_df.to_csv(results_file, index=False)
                logger.info(f"Results saved to {results_file}")
        else:
            print(results_df)
            for units, units_df in results_df.groupby(by="num_units", sort=False):
                for i, sentence in enumerate(units_df["sentence"].values):
                    print(f"{i} [{units}] {sentence}")
                    
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
