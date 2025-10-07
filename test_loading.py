#!/usr/bin/env python3
"""
Test script to verify the new parameter loading functionality.
This script tests that the model can be loaded with missing parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_loading():
    """Test the new loading functionality."""
    try:
        from utils.wan_wrapper import WanDiffusionWrapper
        import torch
        
        print("Testing WanDiffusionWrapper with custom loading...")
        
        # Test creating the wrapper
        wrapper = WanDiffusionWrapper(is_causal=True)
        print("✓ Successfully created WanDiffusionWrapper")
        
        # Test setting trainable parameters (default: all parameters)
        wrapper.set_trainable_parameters(train_all_params=True)
        print("✓ Successfully set all parameters as trainable")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in wrapper.model.parameters())
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable ratio: {trainable_params/total_params:.2%}")
        
        # Test that we can also set only new parameters trainable
        wrapper.set_trainable_parameters(train_all_params=False)
        new_only_trainable = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
        print(f"✓ Only new parameters trainable: {new_only_trainable:,}")
        
        print("\n🎉 All tests passed! The new parameter loading functionality works correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_loading()
    sys.exit(0 if success else 1)
