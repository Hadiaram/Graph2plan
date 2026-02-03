"""
View evaluation metrics from output files
"""
import pickle
import json
import sys
from pathlib import Path

def view_json_metrics(json_path):
    """View metrics from JSON file"""
    print("\n" + "="*80)
    print(f"METRICS FROM: {json_path}")
    print("="*80)
    
    with open(json_path, 'r') as f:
        content = f.read()
        # The file contains a string representation of a dict, need to evaluate it
        try:
            import math
            # Provide nan, inf in the evaluation context
            metrics = eval(content, {"__builtins__": {}, "nan": float('nan'), "inf": float('inf')})
        except Exception as e1:
            try:
                # Try replacing nan with null for JSON
                content_fixed = content.replace('nan', 'null').replace('inf', 'null')
                metrics = json.loads(content_fixed)
            except Exception as e2:
                print(f"Error parsing metrics: {e1}")
                print(f"Fallback error: {e2}")
                print(f"\nRaw content:\n{content}")
                return
    
    print("\n📊 EVALUATION METRICS:\n")
    
    import math
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isnan(value):
                print(f"  {key:20s}: NaN (not computed)")
            elif math.isinf(value):
                print(f"  {key:20s}: Inf")
            else:
                print(f"  {key:20s}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"  {key:20s}: {value}")
    
    print("\n" + "="*80)
    
    # Highlight important metrics
    import math
    if 'box_iou' in metrics and not math.isnan(metrics['box_iou']):
        print(f"\n🎯 Box IoU (Predicted):      {metrics['box_iou']:.4f} ({metrics['box_iou']*100:.2f}%)")
    if 'box_refine_iou' in metrics and not math.isnan(metrics['box_refine_iou']):
        print(f"🎯 Box IoU (Refined):        {metrics['box_refine_iou']:.4f} ({metrics['box_refine_iou']*100:.2f}%)")
    if 'gene_iou' in metrics and not math.isnan(metrics['gene_iou']):
        print(f"🎯 Layout IoU:               {metrics['gene_iou']:.4f} ({metrics['gene_iou']*100:.2f}%)")
    if 'gene_acc' in metrics and not math.isnan(metrics['gene_acc']):
        print(f"📈 Layout Accuracy (no bg):  {metrics['gene_acc']:.4f} ({metrics['gene_acc']*100:.2f}%)")
    if 'gene_acc_all' in metrics and not math.isnan(metrics['gene_acc_all']):
        print(f"📈 Layout Accuracy (all):    {metrics['gene_acc_all']:.4f} ({metrics['gene_acc_all']*100:.2f}%)")
    
    print("="*80 + "\n")


def view_pkl_output(pkl_path):
    """View output from PKL file"""
    print("\n" + "="*80)
    print(f"OUTPUT FROM: {pkl_path}")
    print("="*80)
    
    with open(pkl_path, 'rb') as f:
        output = pickle.load(f)
    
    print(f"\nOutput type: {type(output)}")
    
    if isinstance(output, dict):
        print(f"Number of samples: {len(output)}")
        print(f"Keys in output: {list(output.keys())[:10]}...")
        
        # Show first sample
        first_key = list(output.keys())[0]
        first_sample = output[first_key]
        print(f"\nFirst sample ({first_key}):")
        print(f"  Type: {type(first_sample)}")
        if isinstance(first_sample, dict):
            print(f"  Keys: {list(first_sample.keys())}")
            for k, v in first_sample.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"    {k}: {type(v)}")
    
    print("\n" + "="*80 + "\n")


def main():
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Find most recent metrics file
        experiment_dir = Path('../experiment')
        
        # Look for JSON files first (metrics)
        json_files = list(experiment_dir.rglob('output_*_metrics.json'))
        if json_files:
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            file_path = json_files[0]
            print(f"Found most recent metrics file: {file_path}")
        else:
            # Look for PKL files
            pkl_files = list(experiment_dir.rglob('output_*.pkl'))
            if pkl_files:
                pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                file_path = pkl_files[0]
                print(f"Found most recent output file: {file_path}")
            else:
                print("No output files found!")
                print("Usage: python view_metrics.py [path_to_file]")
                return
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    # Determine file type and view accordingly
    if file_path.suffix == '.json':
        view_json_metrics(file_path)
    elif file_path.suffix == '.pkl':
        view_pkl_output(file_path)
    else:
        print(f"Unknown file type: {file_path.suffix}")


if __name__ == '__main__':
    main()
