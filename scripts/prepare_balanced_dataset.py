#!/usr/bin/env python3
"""
Prepare Balanced Training Dataset

Strategy: Sample equal numbers from two types of .asm files:
- {id}_rocket.asm
- {id}_rlrtl_rocket.asm

This ensures the dataset is balanced across different fuzzer variants.
"""

import re
import os
import json
import pickle
import glob
import random
from typing import List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainingSample:
    """Structured training sample with metadata"""
    sample_id: str
    source_file: str
    file_type: str          # 'rocket' or 'rlrtl_rocket'
    block_id: int
    instructions: List[str]
    num_instructions: int
    is_truncated: bool
    base_address: str


class BasicBlock:
    """Represents a basic block"""
    def __init__(self, block_id: int, base_address: str = ''):
        self.block_id = block_id
        self.base_address = base_address
        self.instructions = []
    
    @property
    def num_instructions(self):
        return len(self.instructions)


class BalancedDatasetBuilder:
    """Build balanced dataset from two types of .asm files"""
    
    def __init__(self, max_instructions: int = 150, min_instructions: int = 3):
        self.max_instructions = max_instructions
        self.min_instructions = min_instructions
    
    def parse_asm_file(self, filepath: str) -> List[BasicBlock]:
        """Parse .asm file and extract basic blocks"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        blocks = []
        current_block = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Basic block header
            bb_match = re.match(r'\[Basic Block (\d+)\]', line)
            if bb_match:
                if current_block is not None:
                    blocks.append(current_block)
                
                block_id = int(bb_match.group(1))
                current_block = BasicBlock(block_id)
                continue
            
            # Base address
            addr_match = re.match(r'Base Address: (0x[0-9a-fA-F]+)', line)
            if addr_match and current_block:
                current_block.base_address = addr_match.group(1)
                continue
            
            # Instruction
            inst_match = re.match(r'0x[0-9a-fA-F]+\s+(.+?)\s+#', line)
            if inst_match and current_block is not None:
                instruction = inst_match.group(1).strip()
                current_block.instructions.append(instruction)
                continue
            
            # Stop at data sections
            if line.startswith('[Initial Register Data]') or \
               line.startswith('[Final Block]') or \
               line.startswith('[Random Data Block]'):
                if current_block is not None:
                    blocks.append(current_block)
                break
        
        if current_block is not None:
            blocks.append(current_block)
        
        return blocks
    
    def process_file(self, filepath: str, file_type: str) -> List[TrainingSample]:
        """Process one .asm file"""
        blocks = self.parse_asm_file(filepath)
        
        if len(blocks) <= 2:
            return []
        
        # Filter: remove first and last (templates)
        blocks = blocks[1:-1]
        
        samples = []
        filename = os.path.basename(filepath)
        
        for block in blocks:
            if block.num_instructions < self.min_instructions:
                continue
            
            is_truncated = block.num_instructions > self.max_instructions
            instructions = block.instructions[:self.max_instructions]
            
            sample = TrainingSample(
                sample_id=f"{filename}_block_{block.block_id}",
                source_file=filename,
                file_type=file_type,
                block_id=block.block_id,
                instructions=instructions,
                num_instructions=len(instructions),
                is_truncated=is_truncated,
                base_address=block.base_address
            )
            
            samples.append(sample)
        
        return samples
    
    def build_balanced_dataset(self, input_dir: str, 
                              num_rocket: int = 500, 
                              num_rlrtl: int = 500) -> List[TrainingSample]:
        """
        Build balanced dataset sampling from two file types
        
        Args:
            input_dir: Directory containing .asm files
            num_rocket: Number of {id}_rocket.asm files to sample
            num_rlrtl: Number of {id}_rlrtl_rocket.asm files to sample
        
        Returns:
            List of training samples from both types
        """
        print("=" * 80)
        print("BUILDING BALANCED DATASET")
        print("=" * 80)
        
        # Find all files
        all_files = glob.glob(os.path.join(input_dir, '*.asm'))
        
        # Separate by type
        rocket_files = []
        rlrtl_files = []
        
        for f in all_files:
            filename = os.path.basename(f)
            if re.match(r'\d+_rlrtl_rocket\.asm$', filename):
                rlrtl_files.append(f)
            elif re.match(r'\d+_rocket\.asm$', filename):
                rocket_files.append(f)
        
        print(f"\nFound files:")
        print(f"  {len(rocket_files):,} files matching {{id}}_rocket.asm")
        print(f"  {len(rlrtl_files):,} files matching {{id}}_rlrtl_rocket.asm")
        
        # Sample files
        print(f"\nSampling:")
        print(f"  {num_rocket} files from {{id}}_rocket.asm")
        print(f"  {num_rlrtl} files from {{id}}_rlrtl_rocket.asm")
        
        if len(rocket_files) < num_rocket:
            print(f"  Warning: Only {len(rocket_files)} rocket files available, using all")
            num_rocket = len(rocket_files)
        
        if len(rlrtl_files) < num_rlrtl:
            print(f"  Warning: Only {len(rlrtl_files)} rlrtl files available, using all")
            num_rlrtl = len(rlrtl_files)
        
        # Random sample
        random.seed(42)  # For reproducibility
        sampled_rocket = random.sample(rocket_files, num_rocket)
        sampled_rlrtl = random.sample(rlrtl_files, num_rlrtl)
        
        print(f"\nProcessing {num_rocket + num_rlrtl} selected files...")
        
        # Process files
        all_samples = []
        
        # Process rocket files
        print(f"\n[1/2] Processing {{id}}_rocket.asm files...")
        for i, filepath in enumerate(sampled_rocket):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{num_rocket}")
            
            try:
                samples = self.process_file(filepath, file_type='rocket')
                all_samples.extend(samples)
            except Exception as e:
                print(f"  Warning: Failed {os.path.basename(filepath)}: {e}")
        
        rocket_samples = len(all_samples)
        print(f"  Extracted {rocket_samples:,} samples from rocket files")
        
        # Process rlrtl files
        print(f"\n[2/2] Processing {{id}}_rlrtl_rocket.asm files...")
        for i, filepath in enumerate(sampled_rlrtl):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{num_rlrtl}")
            
            try:
                samples = self.process_file(filepath, file_type='rlrtl_rocket')
                all_samples.extend(samples)
            except Exception as e:
                print(f"  Warning: Failed {os.path.basename(filepath)}: {e}")
        
        rlrtl_samples = len(all_samples) - rocket_samples
        print(f"  Extracted {rlrtl_samples:,} samples from rlrtl files")
        
        print(f"\n✓ Total samples extracted: {len(all_samples):,}")
        print(f"  From rocket files: {rocket_samples:,}")
        print(f"  From rlrtl files: {rlrtl_samples:,}")
        
        # Shuffle to mix both types
        print(f"\n[Shuffling] Mixing samples from both file types...")
        random.shuffle(all_samples)
        print(f"  ✓ Samples shuffled")
        
        return all_samples
    
    def save_dataset(self, samples: List[TrainingSample], output_prefix: str):
        """Save dataset in multiple formats"""
        print("\n" + "=" * 80)
        print("SAVING DATASET")
        print("=" * 80)
        
        # JSONL
        jsonl_path = f'{output_prefix}.jsonl'
        with open(jsonl_path, 'w') as f:
            for sample in samples:
                json.dump(asdict(sample), f)
                f.write('\n')
        print(f"\n✓ JSONL: {jsonl_path} ({os.path.getsize(jsonl_path)/1024/1024:.1f} MB)")
        
        # Pickle
        pkl_path = f'{output_prefix}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"✓ Pickle: {pkl_path} ({os.path.getsize(pkl_path)/1024/1024:.1f} MB)")
        
        # Metadata
        rocket_samples = [s for s in samples if s.file_type == 'rocket']
        rlrtl_samples = [s for s in samples if s.file_type == 'rlrtl_rocket']
        
        metadata = {
            'total_samples': len(samples),
            'rocket_samples': len(rocket_samples),
            'rlrtl_samples': len(rlrtl_samples),
            'max_instructions': self.max_instructions,
            'min_instructions': self.min_instructions,
            'avg_instructions': sum(s.num_instructions for s in samples) / len(samples),
            'truncated_samples': sum(1 for s in samples if s.is_truncated),
            'unique_source_files': len(set(s.source_file for s in samples)),
        }
        
        metadata_path = f'{output_prefix}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata: {metadata_path}")
        
        return metadata


def print_detailed_statistics(samples: List[TrainingSample]):
    """Print detailed statistics"""
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    
    # Overall stats
    print(f"\nTotal samples: {len(samples):,}")
    
    # By file type
    rocket_samples = [s for s in samples if s.file_type == 'rocket']
    rlrtl_samples = [s for s in samples if s.file_type == 'rlrtl_rocket']
    
    print(f"\nBy file type:")
    print(f"  rocket:        {len(rocket_samples):,} samples ({len(rocket_samples)/len(samples)*100:.1f}%)")
    print(f"  rlrtl_rocket:  {len(rlrtl_samples):,} samples ({len(rlrtl_samples)/len(samples)*100:.1f}%)")
    
    # Instruction count stats
    all_inst_counts = [s.num_instructions for s in samples]
    rocket_inst_counts = [s.num_instructions for s in rocket_samples]
    rlrtl_inst_counts = [s.num_instructions for s in rlrtl_samples]
    
    print(f"\nInstruction counts (all):")
    print(f"  Mean: {sum(all_inst_counts) / len(all_inst_counts):.1f}")
    print(f"  Min: {min(all_inst_counts)}")
    print(f"  Max: {max(all_inst_counts)}")
    
    print(f"\nInstruction counts (rocket):")
    if rocket_inst_counts:
        print(f"  Mean: {sum(rocket_inst_counts) / len(rocket_inst_counts):.1f}")
    
    print(f"\nInstruction counts (rlrtl_rocket):")
    if rlrtl_inst_counts:
        print(f"  Mean: {sum(rlrtl_inst_counts) / len(rlrtl_inst_counts):.1f}")
    
    # Distribution
    print(f"\nInstruction count distribution:")
    bins = [(0, 10), (10, 20), (20, 50), (50, 100), (100, 150)]
    for low, high in bins:
        count = sum(1 for c in all_inst_counts if low <= c < high)
        print(f"  {low:3d}-{high:3d}: {count:6d} samples ({count/len(all_inst_counts)*100:5.1f}%)")
    
    # Total
    total_insts = sum(all_inst_counts)
    print(f"\nTotal instructions: {total_insts:,}")
    print(f"Estimated size: ~{total_insts * 20 / 1024 / 1024:.1f} MB")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build balanced training dataset')
    parser.add_argument('--input-dir', required=True, help='Input directory with .asm files')
    parser.add_argument('--output', required=True, help='Output prefix for dataset files')
    parser.add_argument('--num-rocket', type=int, default=500, 
                       help='Number of {id}_rocket.asm files to sample')
    parser.add_argument('--num-rlrtl', type=int, default=500,
                       help='Number of {id}_rlrtl_rocket.asm files to sample')
    parser.add_argument('--max-inst', type=int, default=150,
                       help='Maximum instructions per sample')
    parser.add_argument('--min-inst', type=int, default=3,
                       help='Minimum instructions per sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Build dataset
    builder = BalancedDatasetBuilder(
        max_instructions=args.max_inst,
        min_instructions=args.min_inst
    )
    
    samples = builder.build_balanced_dataset(
        input_dir=args.input_dir,
        num_rocket=args.num_rocket,
        num_rlrtl=args.num_rlrtl
    )
    
    if not samples:
        print("Error: No samples extracted!")
        return
    
    # Print statistics
    print_detailed_statistics(samples)
    
    # Save dataset
    metadata = builder.save_dataset(samples, args.output)
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET READY")
    print("=" * 80)
    print(f"""
Created balanced dataset with {len(samples):,} samples:
  - From {args.num_rocket} rocket files
  - From {args.num_rlrtl} rlrtl_rocket files
  - Mixed and shuffled

Files generated:
  - {args.output}.jsonl (JSONL format)
  - {args.output}.pkl (Pickle format)
  - {args.output}_metadata.json (Dataset info)

Next step - Train model:
  from lightweight_encoder import train_lightweight_encoder
  import pickle
  
  with open('{args.output}.pkl', 'rb') as f:
      samples = pickle.load(f)
  
  instructions = ['\\n'.join(s.instructions) for s in samples]
  
  model, tokenizer = train_lightweight_encoder(
      testcases=instructions,
      output_dir='models',
      epochs=50,
      batch_size=32
  )
    """)


if __name__ == '__main__':
    main()

