#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class ExpertiseResult:
    """Data class for storing neuron expertise analysis results"""
    concept: str
    concept_group: str
    neuron_metrics: Dict[str, Dict[str, float]]  # Metrics for each neuron
    layer_info: Dict[str, Dict[str, float]]     # Statistics for each layer
    
    def __post_init__(self):
        self.df = self._create_dataframe()
        
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame format"""
        records = []
        for neuron_id, metrics in self.neuron_metrics.items():
            layer_name = neuron_id.split('_')[0] + '_' + neuron_id.split('_')[1]  # Get full layer name
            neuron_idx = int(neuron_id.split('_')[2])  # Get neuron index
            record = {
                'neuron_id': neuron_id,
                'layer': layer_name,
                'neuron_idx': neuron_idx,
                'ap': metrics['ap'],
                'on_p50': metrics['on_value'],
                'off_p50': metrics['off_value']
            }
            records.append(record)
        return pd.DataFrame(records)
    
    def save(self, save_dir: Path) -> None:
        """Save results to files"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(save_dir / "expertise.csv", index=False)
        
        extra_info = {
            "layer_stats": self.layer_info,
            "concept": self.concept,
            "concept_group": self.concept_group
        }
        with open(save_dir / "expertise_info.json", 'w') as f:
            json.dump(extra_info, f, indent=2, cls=NumpyEncoder)
    
    @classmethod
    def load(cls, load_dir: Path) -> 'ExpertiseResult':
        """Load results from files"""
        df = pd.read_csv(load_dir / "expertise.csv")
        with open(load_dir / "expertise_info.json", 'r') as f:
            info = json.load(f)
            
        # Rebuild neuron metrics dictionary
        neuron_metrics = {}
        for _, row in df.iterrows():
            neuron_id = f"{row['layer']}_{row['neuron_idx']}"
            neuron_metrics[neuron_id] = {
                'ap': row['ap'],
                'on_value': row['on_p50'],
                'off_value': row['off_p50']
            }
            
        return cls(
            concept=info['concept'],
            concept_group=info['concept_group'],
            neuron_metrics=neuron_metrics,
            layer_info=info['layer_stats']
        )

def compute_neuron_metrics(
    responses: np.ndarray,
    labels: np.ndarray,
    neuron_idx: int
) -> Dict[str, float]:
    """Compute expertise metrics for a single neuron"""
    # Get positive and negative responses
    pos_responses = responses[labels == 1, neuron_idx]
    neg_responses = responses[labels == 0, neuron_idx]
    
    # Compute AP (Average Precision)
    sorted_indices = np.argsort(-responses[:, neuron_idx])
    sorted_labels = labels[sorted_indices]
    
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    ap = np.mean(precision[sorted_labels == 1])
    
    # Compute ON and OFF values
    on_value = float(np.median(pos_responses))
    off_value = float(np.median(neg_responses))
    
    return {
        'ap': float(ap),
        'on_value': on_value,
        'off_value': off_value
    }

def analyze_concept_expertise(
    concept_dir: Path,
    concept_group: str,
    concept: str,
    model_name: str,
) -> Optional[ExpertiseResult]:
    """Analyze expertise for a specific concept"""
    try:
        # Check if results exist
        expertise_dir = concept_dir / model_name / "expertise"
        if (expertise_dir / "expertise.csv").exists():
            logger.info(f"Skipping {concept_group}/{concept}, expertise analysis already done")
            return None
            
        # Load response
        responses_dir = concept_dir / "responses"
        if not responses_dir.exists():
            logger.warning(f"Response data not found: {responses_dir}")
            return None
            
        # Read all responses
        all_responses = []
        labels = None
        layer_files = sorted(responses_dir.glob("layer_*.npy"))
        
        if not layer_files:
            logger.warning(f"No layer response files found in {responses_dir}")
            return None
            
        # Validate layer file count
        expected_layers = 24  # GPT2-medium has 24 layers
        actual_layers = len(layer_files)
        if actual_layers != expected_layers:
            logger.error(f"Incorrect number of layer files: expected {expected_layers}, got {actual_layers}")
            logger.error(f"Found layer files: {[f.stem for f in layer_files]}")
            return None
            
        # Validatefile names
        expected_layer_names = {f"layer_{i}" for i in range(expected_layers)}
        actual_layer_names = {f.stem for f in layer_files}
        if expected_layer_names != actual_layer_names:
            logger.error(f"Incorrect layer file names")
            logger.error(f"Expected: {sorted(expected_layer_names)}")
            logger.error(f"Got: {sorted(actual_layer_names)}")
            return None
            
        # Read responses for each layer
        for layer_file in layer_files:
            layer_responses = np.load(layer_file)
            if layer_responses.shape[1] != 1024:  # GPT2-medium hidden state dimension
                logger.error(f"Layer {layer_file.stem} has incorrect hidden state dimension: expected 1024, got {layer_responses.shape[1]}")
                return None
                
            if labels is None:
                # Get labels from file name
                labels = np.zeros(len(layer_responses))
                labels[:len(layer_responses)//2] = 1  # First half is positive examples
            all_responses.append(layer_responses)
            
        # Concatenate all layer responses
        responses = np.concatenate(all_responses, axis=1)
        
        # Compute metrics for each neuron
        neuron_metrics = {}
        layer_info = {}
        
        # Process each layer
        for layer_idx in range(expected_layers):
            layer_name = f"layer_{layer_idx}"
            layer_start_idx = layer_idx * 1024
            layer_end_idx = (layer_idx + 1) * 1024
            
            # Compute metrics for all neurons in this layer
            for neuron_idx in range(1024):
                global_idx = layer_start_idx + neuron_idx
                metrics = compute_neuron_metrics(responses, labels, global_idx)
                neuron_metrics[f"{layer_name}_{neuron_idx}"] = {
                    **metrics,
                    "layer": layer_name,
                    "neuron_idx": neuron_idx
                }
            
            # Collect layer-level statistics
            layer_metrics = [m['ap'] for m in neuron_metrics.values() 
                           if m['layer'] == layer_name and m['ap'] > 0]
            if layer_metrics:
                layer_info[layer_name] = {
                    'mean_ap': float(np.mean(layer_metrics)),
                    'max_ap': float(np.max(layer_metrics)),
                    'mean_on_value': float(np.mean([m['on_value'] for m in neuron_metrics.values() 
                                                  if m['layer'] == layer_name]))
                }
        
        # Create result object
        result = ExpertiseResult(
            concept=concept,
            concept_group=concept_group,
            neuron_metrics=neuron_metrics,
            layer_info=layer_info
        )
        
        # Save results
        result.save(expertise_dir)
        logger.info(f"Completed expertise analysis for {concept_group}/{concept}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing concept {concept_group}/{concept}: {str(e)}")
        return None

def plot_expertise_results(
    result: ExpertiseResult,
    save_dir: Path,
    top_k: int = 10,
    show_plots: bool = False
) -> None:
    """Plot visualization of expertise analysis results"""
    try:
        # Create plots directory
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot AP distribution scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(result.df['ap'], result.df['on_p50'], alpha=0.1)
        plt.xlabel('Average Precision (AP)')
        plt.ylabel('ON Value')
        plt.title(f'AP vs ON Value ({result.concept_group}/{result.concept})')
        plt.savefig(plots_dir / 'ap_vs_on.png')
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot top-k neurons AP per layer
        plt.figure(figsize=(12, 6))
        for layer in sorted(result.df['layer'].unique()):
            layer_data = result.df[result.df['layer'] == layer]
            top_k_ap = layer_data.nlargest(top_k, 'ap')['ap']
            plt.plot(range(len(top_k_ap)), top_k_ap, label=layer)
            
        plt.xlabel('Neuron Rank')
        plt.ylabel('AP')
        plt.title(f'Top-{top_k} Neurons AP per Layer')
        plt.legend()
        plt.savefig(plots_dir / f'top_{top_k}_ap_per_layer.png')
        if show_plots:
            plt.show()
        plt.close()
        
        logger.info(f"Saved visualization results to {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Compute neuron expertise for concepts")
    parser.add_argument("--root-dir", required=True, type=Path, help="Root directory containing response data")
    parser.add_argument("--assets-dir", required=True, type=Path, help="Assets directory containing concept lists")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--concepts", help="List of concepts to analyze (comma-separated)")
    parser.add_argument("--top-k", type=int, default=10, help="Show top-k neurons per layer")
    parser.add_argument("--show", action="store_true", help="Show plots")
    args = parser.parse_args()
    
    try:
        # Get concept list
        if args.concepts:
            concepts = args.concepts.split(',')
            concept_df = pd.DataFrame([
                {'concept': c.split('/')[1], 'group': c.split('/')[0]}
                for c in concepts
            ])
        else:
            # Read concept lists from assets directory
            concept_lists = []
            for dataset_dir in args.assets_dir.iterdir():
                if dataset_dir.is_dir() and (dataset_dir / "concept_list.csv").exists():
                    df = pd.read_csv(dataset_dir / "concept_list.csv")
                    concept_lists.append(df)
            
            if not concept_lists:
                raise FileNotFoundError(f"No concept_list.csv files found in {args.assets_dir}")
                
            concept_df = pd.concat(concept_lists, ignore_index=True)
            
        logger.info(f"Found {len(concept_df)} concepts to analyze")
        
        # Analyze each concept
        for _, row in tqdm(concept_df.iterrows(), total=len(concept_df)):
            concept_dir = args.root_dir / row['group'] / row['concept']
            
            # Compute expertise
            result = analyze_concept_expertise(
                concept_dir=concept_dir,
                concept=row['concept'],
                concept_group=row['group'],
                model_name=args.model_name
            )
            
            if result:
                # Plot results
                plot_expertise_results(
                    result=result,
                    save_dir=concept_dir / args.model_name / "expertise",
                    top_k=args.top_k,
                    show_plots=args.show
                )
                
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
