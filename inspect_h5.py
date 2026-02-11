#!/usr/bin/env python3
"""Inspect .h5 model to understand architecture and training"""
import h5py
import json
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else 'model_float32.h5'

f = h5py.File(filepath, 'r')

# Print top-level keys
print('=== TOP-LEVEL KEYS ===')
for key in f.keys():
    print(f'  {key}')

# Try to get model config
print('\n=== MODEL CONFIG ===')
if 'model_config' in f.attrs:
    config = f.attrs['model_config']
    if isinstance(config, bytes):
        config = config.decode('utf-8')
    cfg = json.loads(config)
    
    # Model class name
    class_name = cfg.get('class_name', 'Unknown')
    print(f'Model type: {class_name}')
    
    # Get layers info
    if 'config' in cfg:
        model_cfg = cfg['config']
        model_name = model_cfg.get('name', 'unknown')
        print(f'Model name: {model_name}')
        
        if 'layers' in model_cfg:
            layers = model_cfg['layers']
            print(f'Total layers: {len(layers)}')
            
            # Show first 15 layers
            print('\nFirst 15 layers:')
            for l in layers[:15]:
                cn = l.get('class_name', '?')
                lconfig = l.get('config', {})
                name = lconfig.get('name', l.get('name', '?'))
                
                # Extra info per layer type
                extra = ''
                if cn == 'Conv2D':
                    filters = lconfig.get('filters', '?')
                    ks = lconfig.get('kernel_size', '?')
                    act = lconfig.get('activation', 'none')
                    extra = f'filters={filters} kernel={ks} act={act}'
                elif cn == 'Dense':
                    units = lconfig.get('units', '?')
                    act = lconfig.get('activation', 'none')
                    extra = f'units={units} act={act}'
                elif cn == 'InputLayer':
                    batch_shape = lconfig.get('batch_input_shape', lconfig.get('batch_shape', '?'))
                    extra = f'shape={batch_shape}'
                elif cn == 'DepthwiseConv2D':
                    ks = lconfig.get('kernel_size', '?')
                    extra = f'kernel={ks}'
                elif cn in ('MaxPooling2D', 'AveragePooling2D'):
                    pool = lconfig.get('pool_size', '?')
                    extra = f'pool={pool}'
                elif cn == 'BatchNormalization':
                    extra = 'BN'
                elif cn == 'LeakyReLU':
                    alpha = lconfig.get('alpha', lconfig.get('negative_slope', '?'))
                    extra = f'alpha={alpha}'
                elif cn == 'Activation':
                    act = lconfig.get('activation', '?')
                    extra = f'activation={act}'
                
                if extra:
                    print(f'  {name:40s} ({cn}) [{extra}]')
                else:
                    print(f'  {name:40s} ({cn})')
            
            if len(layers) > 30:
                print(f'\n  ... ({len(layers)-30} layers omitted) ...\n')
            
            # Show last 15 layers
            print('Last 15 layers:')
            for l in layers[-15:]:
                cn = l.get('class_name', '?')
                lconfig = l.get('config', {})
                name = lconfig.get('name', l.get('name', '?'))
                
                extra = ''
                if cn == 'Conv2D':
                    filters = lconfig.get('filters', '?')
                    ks = lconfig.get('kernel_size', '?')
                    act = lconfig.get('activation', 'none')
                    extra = f'filters={filters} kernel={ks} act={act}'
                elif cn == 'Dense':
                    units = lconfig.get('units', '?')
                    act = lconfig.get('activation', 'none')
                    extra = f'units={units} act={act}'
                elif cn == 'LeakyReLU':
                    alpha = lconfig.get('alpha', lconfig.get('negative_slope', '?'))
                    extra = f'alpha={alpha}'
                elif cn == 'Concatenate':
                    axis = lconfig.get('axis', '?')
                    extra = f'axis={axis}'
                
                if extra:
                    print(f'  {name:40s} ({cn}) [{extra}]')
                else:
                    print(f'  {name:40s} ({cn})')
            
            # Count layer types
            print('\n=== LAYER TYPE SUMMARY ===')
            type_count = {}
            activations_used = set()
            for l in layers:
                cn = l.get('class_name', '?')
                type_count[cn] = type_count.get(cn, 0) + 1
                lconfig = l.get('config', {})
                if cn == 'Conv2D':
                    act = lconfig.get('activation', 'linear')
                    if act != 'linear':
                        activations_used.add(act)
                elif cn == 'Dense':
                    act = lconfig.get('activation', 'linear')
                    if act != 'linear':
                        activations_used.add(act)
                elif cn == 'Activation':
                    activations_used.add(lconfig.get('activation', '?'))
                elif cn == 'LeakyReLU':
                    activations_used.add('leaky_relu')
                elif cn == 'ReLU':
                    activations_used.add('relu')
            
            for cn, count in sorted(type_count.items(), key=lambda x: -x[1]):
                print(f'  {cn:30s}: {count}')
            
            print(f'\n  Activations used: {activations_used}')
            
            # Check for NPU-incompatible activations
            bad_acts = activations_used & {'silu', 'swish', 'mish', 'gelu', 'elu', 'selu'}
            good_acts = activations_used & {'relu', 'relu6', 'leaky_relu', 'sigmoid', 'tanh', 'softmax', 'linear'}
            
            if bad_acts:
                print(f'\n  ⚠ NPU-INCOMPATIBLE activations: {bad_acts}')
                print(f'    These need to be replaced before Vela conversion!')
            else:
                print(f'\n  ✓ All activations are NPU-compatible!')

# Check training config
if 'training_config' in f.attrs:
    tc = f.attrs['training_config']
    if isinstance(tc, bytes):
        tc = tc.decode('utf-8')
    tc_dict = json.loads(tc)
    print(f'\n=== TRAINING CONFIG ===')
    if 'optimizer_config' in tc_dict:
        opt = tc_dict['optimizer_config']
        opt_name = opt.get('class_name', '?')
        print(f'Optimizer: {opt_name}')
        if 'config' in opt:
            lr = opt['config'].get('learning_rate', opt['config'].get('lr', '?'))
            print(f'Learning rate: {lr}')
    if 'loss' in tc_dict:
        loss = tc_dict['loss']
        if isinstance(loss, dict):
            print(f'Loss functions: {list(loss.keys())}')
        else:
            print(f'Loss: {loss}')

# Backend
if 'keras_version' in f.attrs:
    kv = f.attrs['keras_version']
    if isinstance(kv, bytes):
        kv = kv.decode('utf-8')
    print(f'\nKeras version: {kv}')
if 'backend' in f.attrs:
    be = f.attrs['backend']
    if isinstance(be, bytes):
        be = be.decode('utf-8')
    print(f'Backend: {be}')

f.close()
