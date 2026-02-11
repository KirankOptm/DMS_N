#!/usr/bin/env python3
"""
TFLite Model Operator Analyzer for Ethos-U55 (Vela) Compatibility
=================================================================
Parses TFLite flatbuffer directly to extract ALL operator types,
then checks each against the Vela/Ethos-U55 supported operator list.

Works WITHOUT TensorFlow - uses only 'tflite' and 'flatbuffers' packages.

Usage: py -3 analyze_tflite_ops.py <model.tflite>
"""

import sys
import os
import struct

# ============================================================
# TFLite BuiltinOperator enum (from schema_generated.h)
# Complete list up to TFLite schema v3.18
# ============================================================
BUILTIN_OPERATOR_NAMES = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP",
    8: "FLOOR",
    9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP",
    11: "L2_NORMALIZATION",
    12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION",
    14: "LOGISTIC",   # Sigmoid
    15: "LSH_PROJECTION",
    16: "LSTM",
    17: "MAX_POOL_2D",
    18: "MUL",
    19: "RELU",
    20: "RELU_N1_TO_1",
    21: "RELU6",
    22: "RESHAPE",
    23: "RESIZE_BILINEAR",
    24: "RNN",
    25: "SOFTMAX",
    26: "SPACE_TO_DEPTH",
    27: "SVDF",
    28: "TANH",
    29: "CONCAT_EMBEDDINGS",
    30: "SKIP_GRAM",
    31: "CALL",
    32: "CUSTOM",
    33: "EMBEDDING_LOOKUP_SPARSE",
    34: "PAD",
    35: "UNIDIRECTIONAL_SEQUENCE_RNN",
    36: "GATHER",
    37: "BATCH_TO_SPACE_ND",
    38: "SPACE_TO_BATCH_ND",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    42: "DIV",
    43: "SQUEEZE",
    44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
    45: "STRIDED_SLICE",
    46: "BIDIRECTIONAL_SEQUENCE_RNN",
    47: "EXP",
    48: "TOPK_V2",
    49: "SPLIT",
    50: "LOG_SOFTMAX",
    51: "DELEGATE",
    52: "BIDIRECTIONAL_SEQUENCE_LSTM",
    53: "CAST",
    54: "PRELU",
    55: "MAXIMUM",
    56: "ARG_MAX",
    57: "MINIMUM",
    58: "LESS",
    59: "NEG",
    60: "PADV2",
    61: "GREATER",
    62: "GREATER_EQUAL",
    63: "LESS_EQUAL",
    64: "SELECT",
    65: "SLICE",
    66: "SIN",
    67: "TRANSPOSE_CONV",
    68: "SPARSE_TO_DENSE",
    69: "TILE",
    70: "EXPAND_DIMS",
    71: "EQUAL",
    72: "NOT_EQUAL",
    73: "LOG",
    74: "SUM",
    75: "SQRT",
    76: "RSQRT",
    77: "SHAPE",
    78: "POW",
    79: "ARG_MIN",
    80: "FAKE_QUANT",
    81: "REDUCE_PROD",
    82: "REDUCE_MAX",
    83: "PACK",
    84: "LOGICAL_OR",
    85: "ONE_HOT",
    86: "LOGICAL_AND",
    87: "LOGICAL_NOT",
    88: "UNPACK",
    89: "REDUCE_MIN",
    90: "FLOOR_DIV",
    91: "REDUCE_ANY",
    92: "SQUARE",
    93: "ZEROS_LIKE",
    94: "FILL",
    95: "FLOOR_MOD",
    96: "RANGE",
    97: "RESIZE_NEAREST_NEIGHBOR",
    98: "LEAKY_RELU",
    99: "SQUARED_DIFFERENCE",
    100: "MIRROR_PAD",
    101: "ABS",
    102: "SPLIT_V",
    103: "UNIQUE",
    104: "CEIL",
    105: "REVERSE_V2",
    106: "ADD_N",
    107: "GATHER_ND",
    108: "COS",
    109: "WHERE",
    110: "RANK",
    111: "ELU",
    112: "REVERSE_SEQUENCE",
    113: "MATRIX_DIAG",
    114: "QUANTIZE",
    115: "MATRIX_SET_DIAG",
    116: "ROUND",
    117: "HARD_SWISH",
    118: "IF",
    119: "WHILE",
    120: "NON_MAX_SUPPRESSION_V4",
    121: "NON_MAX_SUPPRESSION_V5",
    122: "SCATTER_ND",
    123: "SELECT_V2",
    124: "DENSIFY",
    125: "SEGMENT_SUM",
    126: "BATCH_MATMUL",
    127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
    128: "CUMSUM",
    129: "CALL_ONCE",
    130: "BROADCAST_TO",
    131: "RFFT2D",
    132: "CONV_3D",
    133: "IMAG",
    134: "REAL",
    135: "COMPLEX_ABS",
    136: "HASHTABLE",
    137: "HASHTABLE_FIND",
    138: "HASHTABLE_IMPORT",
    139: "HASHTABLE_SIZE",
    140: "REDUCE_ALL",
    141: "CONV_3D_TRANSPOSE",
    142: "VAR_HANDLE",
    143: "READ_VARIABLE",
    144: "ASSIGN_VARIABLE",
    145: "BROADCAST_ARGS",
    146: "RANDOM_STANDARD_NORMAL",
    147: "BUCKETIZE",
    148: "RANDOM_UNIFORM",
    149: "MULTINOMIAL",
    150: "GELU",
    151: "DYNAMIC_UPDATE_SLICE",
    152: "RELU_0_TO_1",
    153: "UNSORTED_SEGMENT_PROD",
    154: "UNSORTED_SEGMENT_MAX",
    155: "UNSORTED_SEGMENT_SUM",
    156: "ATAN2",
    157: "UNSORTED_SEGMENT_MIN",
    158: "SIGN",
    159: "BITCAST",
    160: "BITWISE_XOR",
    161: "RIGHT_SHIFT",
    162: "STABLEHLO_LOGISTIC",
    163: "STABLEHLO_ADD",
    164: "STABLEHLO_DIVIDE",
    165: "STABLEHLO_MULTIPLY",
    166: "STABLEHLO_MAXIMUM",
    167: "STABLEHLO_RESHAPE",
    168: "STABLEHLO_CLAMP",
    169: "STABLEHLO_CONCATENATE",
    170: "STABLEHLO_BROADCAST_IN_DIM",
    171: "STABLEHLO_CONVOLUTION",
    172: "STABLEHLO_SLICE",
    173: "STABLEHLO_CUSTOM_CALL",
    174: "STABLEHLO_REDUCE",
    175: "STABLEHLO_ABS",
    176: "STABLEHLO_AND",
    177: "STABLEHLO_COSINE",
    178: "STABLEHLO_EXPONENTIAL",
    179: "STABLEHLO_FLOOR",
    180: "STABLEHLO_LOG",
    181: "STABLEHLO_MINIMUM",
    182: "STABLEHLO_NEGATE",
    183: "STABLEHLO_OR",
    184: "STABLEHLO_POWER",
    185: "STABLEHLO_REMAINDER",
    186: "STABLEHLO_RSQRT",
    187: "STABLEHLO_SELECT",
    188: "STABLEHLO_SUBTRACT",
    189: "STABLEHLO_TANH",
    190: "STABLEHLO_SCATTER",
    191: "STABLEHLO_COMPARE",
    192: "STABLEHLO_CONVERT",
    193: "STABLEHLO_DYNAMIC_SLICE",
    194: "STABLEHLO_DYNAMIC_UPDATE_SLICE",
    195: "STABLEHLO_PAD",
    196: "STABLEHLO_IOTA",
    197: "STABLEHLO_DOT_GENERAL",
    198: "STABLEHLO_REDUCE_WINDOW",
    199: "STABLEHLO_SORT",
    200: "STABLEHLO_WHILE",
    201: "STABLEHLO_GATHER",
    202: "STABLEHLO_TRANSPOSE",
    203: "DILATE",
    204: "STABLEHLO_RNG_BIT_GENERATOR",
    205: "REDUCE_WINDOW",
}


# ============================================================
# Ethos-U55 Vela Supported Operators (from Vela documentation)
# These are the TFLite operators that Vela can accelerate
# ============================================================
VELA_SUPPORTED_OPS = {
    "ADD",
    "AVERAGE_POOL_2D",
    "CONCATENATION",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "DEQUANTIZE",
    "EXPAND_DIMS",
    "FULLY_CONNECTED",
    "HARD_SWISH",
    "LEAKY_RELU",
    "LOGISTIC",          # Sigmoid
    "MAX_POOL_2D",
    "MEAN",
    "MINIMUM",
    "MAXIMUM",
    "MUL",
    "PAD",
    "PADV2",
    "QUANTIZE",
    "RELU",
    "RELU6",
    "RELU_N1_TO_1",
    "RESHAPE",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
    "SHAPE",
    "SLICE",
    "SOFTMAX",
    "SPACE_TO_DEPTH",
    "SPLIT",
    "SPLIT_V",
    "SQUEEZE",
    "STRIDED_SLICE",
    "SUB",
    "TANH",
    "TRANSPOSE",
    "TRANSPOSE_CONV",
    "UNPACK",
    "PACK",
    "ABS",
    "PRELU",
}

# Operators that Vela accepts but may fall back to CPU
VELA_PARTIAL_SUPPORT = {
    "BATCH_TO_SPACE_ND",
    "SPACE_TO_BATCH_ND",
    "GATHER",
    "GATHER_ND",
    "ARG_MAX",
    "ARG_MIN",
    "REDUCE_MAX",
    "REDUCE_MIN",
    "EXP",
    "LOG",
    "SQRT",
    "RSQRT",
    "NEG",
    "CAST",
    "FLOOR",
    "CEIL",
    "ROUND",
    "TILE",
    "MIRROR_PAD",
    "SELECT",
    "SELECT_V2",
    "LESS",
    "LESS_EQUAL",
    "GREATER",
    "GREATER_EQUAL",
    "EQUAL",
    "NOT_EQUAL",
    "SQUARE",
    "SUM",
    "REDUCE_PROD",
    "FILL",
    "ZEROS_LIKE",
    "TOPK_V2",
    "NON_MAX_SUPPRESSION_V4",
    "NON_MAX_SUPPRESSION_V5",
}

# Known unsupported ops that break Vela
VELA_UNSUPPORTED_OPS = {
    "CUSTOM",        # Anything custom will NOT run on NPU
    "GELU",
    "ELU",
    "LSTM",
    "RNN",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
    "BIDIRECTIONAL_SEQUENCE_RNN",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "SVDF",
    "EMBEDDING_LOOKUP",
    "HASHTABLE_LOOKUP",
    "L2_NORMALIZATION",
    "LOCAL_RESPONSE_NORMALIZATION",
    "BATCH_MATMUL",
    "CONV_3D",
    "CONV_3D_TRANSPOSE",
    "SEGMENT_SUM",
    "CUMSUM",
    "REVERSE_V2",
    "REVERSE_SEQUENCE",
    "SCATTER_ND",
    "WHERE",
    "RANGE",
    "RANK",
    "UNIQUE",
    "MATRIX_DIAG",
    "MATRIX_SET_DIAG",
    "DIV",
    "FLOOR_DIV",
    "FLOOR_MOD",
    "POW",
    "LOGICAL_AND",
    "LOGICAL_OR",
    "LOGICAL_NOT",
    "ADD_N",
    "SQUARED_DIFFERENCE",
    "ONE_HOT",
    "FAKE_QUANT",
    "DENSIFY",
    "IF",
    "WHILE",
    "CALL",
    "LSH_PROJECTION",
    "SKIP_GRAM",
    "CONCAT_EMBEDDINGS",
    "EMBEDDING_LOOKUP_SPARSE",
    "SPARSE_TO_DENSE",
    "SIN",
    "COS",
    "RFFT2D",
    "IMAG",
    "REAL",
    "COMPLEX_ABS",
    "MULTINOMIAL",
    "RANDOM_STANDARD_NORMAL",
    "RANDOM_UNIFORM",
    "BUCKETIZE",
    "DYNAMIC_UPDATE_SLICE",
    "L2_POOL_2D",
    "DEPTH_TO_SPACE",
    "REDUCE_ANY",
    "REDUCE_ALL",
    "BROADCAST_TO",
    "SIGN",
    "ATAN2",
}


def parse_tflite_model(filepath):
    """Parse TFLite flatbuffer to extract model info"""
    try:
        from tflite.Model import Model
        
        with open(filepath, 'rb') as f:
            buf = f.read()
        
        model = Model.GetRootAs(buf, 0)
        return model, buf
    except Exception as e:
        print(f"Error parsing model: {e}")
        return None, None


def get_operator_info(model):
    """Extract all operator types from the model"""
    operators = []
    
    # Get operator codes (unique operator types used in model)
    op_codes_count = model.OperatorCodesLength()
    op_code_map = {}
    
    for i in range(op_codes_count):
        op_code = model.OperatorCodes(i)
        # BuiltinCode() is the deprecated field (int8), 
        # DeprecatedBuiltinCode() for older models
        # For newer models, use BuiltinCode()
        builtin_code = op_code.BuiltinCode()
        deprecated_code = op_code.DeprecatedBuiltinCode()
        custom_code = op_code.CustomCode()
        
        # If deprecated_code != 127 and builtin_code == 0, use deprecated
        if deprecated_code != 127 and builtin_code == 0:
            actual_code = deprecated_code
        else:
            actual_code = builtin_code
            
        op_name = BUILTIN_OPERATOR_NAMES.get(actual_code, f"UNKNOWN_{actual_code}")
        
        if op_name == "CUSTOM" and custom_code:
            custom_name = custom_code.decode('utf-8') if isinstance(custom_code, bytes) else str(custom_code)
            op_name = f"CUSTOM({custom_name})"
        
        op_code_map[i] = op_name
    
    # Count operator usage per subgraph
    op_usage = {}
    total_ops = 0
    
    for sg_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(sg_idx)
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            op_code_idx = op.OpcodeIndex()
            op_name = op_code_map.get(op_code_idx, f"UNKNOWN_IDX_{op_code_idx}")
            
            if op_name not in op_usage:
                op_usage[op_name] = 0
            op_usage[op_name] += 1
            total_ops += 1
    
    return op_code_map, op_usage, total_ops


def get_tensor_info(model):
    """Extract input/output tensor information"""
    info = {"inputs": [], "outputs": [], "all_tensors": []}
    
    for sg_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(sg_idx)
        
        # Inputs
        for i in range(subgraph.InputsLength()):
            tensor_idx = subgraph.Inputs(i)
            tensor = subgraph.Tensors(tensor_idx)
            
            shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
            dtype_map = {0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 
                        4: "INT64", 5: "STRING", 6: "BOOL", 7: "INT16",
                        8: "COMPLEX64", 9: "INT8", 10: "FLOAT64", 11: "COMPLEX128",
                        12: "UINT64", 13: "RESOURCE", 14: "VARIANT", 15: "UINT32",
                        16: "UINT16", 17: "INT4", 18: "UINT4"}
            dtype = dtype_map.get(tensor.Type(), f"UNKNOWN_{tensor.Type()}")
            
            name = tensor.Name()
            if name:
                name = name.decode('utf-8') if isinstance(name, bytes) else str(name)
            
            quant = tensor.Quantization()
            quant_info = None
            if quant and quant.ScaleLength() > 0:
                scale = quant.Scale(0)
                zp = quant.ZeroPoint(0) if quant.ZeroPointLength() > 0 else 0
                if scale != 0:
                    quant_info = {"scale": scale, "zero_point": zp}
            
            info["inputs"].append({
                "name": name, "shape": shape, "dtype": dtype, 
                "quantized": quant_info is not None, "quant": quant_info
            })
        
        # Outputs
        for i in range(subgraph.OutputsLength()):
            tensor_idx = subgraph.Outputs(i)
            tensor = subgraph.Tensors(tensor_idx)
            
            shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
            dtype = dtype_map.get(tensor.Type(), f"UNKNOWN_{tensor.Type()}")
            
            name = tensor.Name()
            if name:
                name = name.decode('utf-8') if isinstance(name, bytes) else str(name)
            
            quant = tensor.Quantization()
            quant_info = None
            if quant and quant.ScaleLength() > 0:
                scale = quant.Scale(0)
                zp = quant.ZeroPoint(0) if quant.ZeroPointLength() > 0 else 0
                if scale != 0:
                    quant_info = {"scale": scale, "zero_point": zp}
            
            info["outputs"].append({
                "name": name, "shape": shape, "dtype": dtype,
                "quantized": quant_info is not None, "quant": quant_info
            })
        
        # Count all tensors by type
        for i in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(i)
            dtype = dtype_map.get(tensor.Type(), f"UNKNOWN_{tensor.Type()}")
            info["all_tensors"].append(dtype)
    
    return info


def analyze_vela_compatibility(op_usage):
    """Check each operator against Vela supported list"""
    supported = {}
    partial = {}
    unsupported = {}
    unknown = {}
    
    for op_name, count in op_usage.items():
        # Handle CUSTOM ops specially
        base_name = op_name.split("(")[0] if "(" in op_name else op_name
        
        if base_name in VELA_SUPPORTED_OPS:
            supported[op_name] = count
        elif base_name in VELA_PARTIAL_SUPPORT:
            partial[op_name] = count
        elif base_name in VELA_UNSUPPORTED_OPS or base_name == "CUSTOM":
            unsupported[op_name] = count
        else:
            unknown[op_name] = count
    
    return supported, partial, unsupported, unknown


def estimate_mac_ops(model):
    """Rough estimation of MAC operations for Conv2D and DepthwiseConv2D"""
    total_macs = 0
    conv_details = []
    
    for sg_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(sg_idx)
        
        # Build op_code lookup
        op_codes = {}
        for i in range(model.OperatorCodesLength()):
            oc = model.OperatorCodes(i)
            bc = oc.BuiltinCode()
            dc = oc.DeprecatedBuiltinCode()
            if dc != 127 and bc == 0:
                op_codes[i] = dc
            else:
                op_codes[i] = bc
        
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            code = op_codes.get(op.OpcodeIndex(), -1)
            
            if code == 3:  # CONV_2D
                # Get filter tensor (input 1)
                if op.InputsLength() >= 2:
                    filter_idx = op.Inputs(1)
                    ft = subgraph.Tensors(filter_idx)
                    if ft.ShapeLength() == 4:
                        oc = ft.Shape(0)  # output channels
                        kh = ft.Shape(1)
                        kw = ft.Shape(2)
                        ic = ft.Shape(3)  # input channels
                        
                        # Get output tensor
                        if op.OutputsLength() >= 1:
                            out_idx = op.Outputs(0)
                            ot = subgraph.Tensors(out_idx)
                            if ot.ShapeLength() == 4:
                                oh = ot.Shape(1)
                                ow = ot.Shape(2)
                                macs = kh * kw * ic * oc * oh * ow
                                total_macs += macs
                                conv_details.append(f"  Conv2D: {ic}x{kh}x{kw} -> {oc} out:{oh}x{ow} = {macs/1e6:.2f}M MACs")
            
            elif code == 4:  # DEPTHWISE_CONV_2D
                if op.InputsLength() >= 2:
                    filter_idx = op.Inputs(1)
                    ft = subgraph.Tensors(filter_idx)
                    if ft.ShapeLength() == 4:
                        dm = ft.Shape(0)   # depth multiplier (or 1)
                        kh = ft.Shape(1)
                        kw = ft.Shape(2)
                        ic = ft.Shape(3)
                        
                        if op.OutputsLength() >= 1:
                            out_idx = op.Outputs(0)
                            ot = subgraph.Tensors(out_idx)
                            if ot.ShapeLength() == 4:
                                oh = ot.Shape(1)
                                ow = ot.Shape(2)
                                macs = kh * kw * ic * dm * oh * ow
                                total_macs += macs
                                conv_details.append(f"  DWConv: {ic}x{kh}x{kw} dm={dm} out:{oh}x{ow} = {macs/1e6:.2f}M MACs")
            
            elif code == 9:  # FULLY_CONNECTED
                if op.InputsLength() >= 2:
                    filter_idx = op.Inputs(1)
                    ft = subgraph.Tensors(filter_idx)
                    if ft.ShapeLength() == 2:
                        out_features = ft.Shape(0)
                        in_features = ft.Shape(1)
                        macs = in_features * out_features
                        total_macs += macs
                        conv_details.append(f"  FC: {in_features} -> {out_features} = {macs/1e6:.2f}M MACs")
    
    return total_macs, conv_details


def main():
    if len(sys.argv) < 2:
        print("Usage: py -3 analyze_tflite_ops.py <model.tflite> [model2.tflite ...]")
        print("\nAvailable .tflite files in current directory:")
        for f in sorted(os.listdir('.')):
            if f.endswith('.tflite'):
                size = os.path.getsize(f) / 1024
                print(f"  {f} ({size:.1f} KB)")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        print("\n" + "=" * 72)
        print(f"  TFLITE MODEL ANALYSIS: {os.path.basename(filepath)}")
        print("=" * 72)
        
        if not os.path.exists(filepath):
            print(f"  ERROR: File not found: {filepath}")
            continue
        
        file_size = os.path.getsize(filepath)
        print(f"\n  File size: {file_size/1024:.1f} KB ({file_size/1024/1024:.2f} MB)")
        
        # Parse model
        model, buf = parse_tflite_model(filepath)
        if model is None:
            continue
        
        # Get tensor info
        tensor_info = get_tensor_info(model)
        
        print(f"\n{'─'*72}")
        print("  INPUT TENSORS")
        print(f"{'─'*72}")
        for inp in tensor_info["inputs"]:
            q_str = "INT8 quantized" if inp["quantized"] else inp["dtype"]
            print(f"  Name:  {inp['name']}")
            print(f"  Shape: {inp['shape']}")
            print(f"  Type:  {q_str}")
            if inp["quant"]:
                print(f"  Scale: {inp['quant']['scale']:.6f}, ZP: {inp['quant']['zero_point']}")
            print()
        
        print(f"{'─'*72}")
        print("  OUTPUT TENSORS")
        print(f"{'─'*72}")
        for out in tensor_info["outputs"]:
            q_str = "INT8 quantized" if out["quantized"] else out["dtype"]
            print(f"  Name:  {out['name']}")
            print(f"  Shape: {out['shape']}")
            print(f"  Type:  {q_str}")
            if out["quant"]:
                print(f"  Scale: {out['quant']['scale']:.6f}, ZP: {out['quant']['zero_point']}")
            print()
        
        # Quantization summary
        dtype_counts = {}
        for dt in tensor_info["all_tensors"]:
            dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        
        print(f"{'─'*72}")
        print("  TENSOR TYPES (all tensors)")
        print(f"{'─'*72}")
        for dt, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
            pct = count / len(tensor_info["all_tensors"]) * 100
            bar = "█" * int(pct / 2)
            print(f"  {dt:12s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        is_quantized = "INT8" in dtype_counts or "UINT8" in dtype_counts
        float_count = dtype_counts.get("FLOAT32", 0) + dtype_counts.get("FLOAT16", 0)
        int_count = dtype_counts.get("INT8", 0) + dtype_counts.get("UINT8", 0)
        
        # Get operator info
        op_code_map, op_usage, total_ops = get_operator_info(model)
        
        print(f"\n{'─'*72}")
        print(f"  OPERATORS ({total_ops} total operations)")
        print(f"{'─'*72}")
        for op_name, count in sorted(op_usage.items(), key=lambda x: -x[1]):
            pct = count / total_ops * 100
            print(f"  {op_name:35s}: {count:4d} ({pct:5.1f}%)")
        
        # Vela compatibility analysis
        supported, partial, unsupported, unknown = analyze_vela_compatibility(op_usage)
        
        total_supported = sum(supported.values())
        total_partial = sum(partial.values())
        total_unsupported = sum(unsupported.values())
        total_unknown = sum(unknown.values())
        
        npu_pct = (total_supported / total_ops * 100) if total_ops > 0 else 0
        
        print(f"\n{'─'*72}")
        print("  ETHOS-U55 (VELA) COMPATIBILITY ANALYSIS")
        print(f"{'─'*72}")
        
        if supported:
            print(f"\n  ✓ FULLY SUPPORTED on NPU ({total_supported}/{total_ops} ops, {npu_pct:.1f}%):")
            for op, count in sorted(supported.items(), key=lambda x: -x[1]):
                print(f"    ✓ {op:30s} x{count}")
        
        if partial:
            print(f"\n  ~ PARTIAL SUPPORT ({total_partial} ops - may fall back to CPU):")
            for op, count in sorted(partial.items(), key=lambda x: -x[1]):
                print(f"    ~ {op:30s} x{count}")
        
        if unsupported:
            print(f"\n  ✗ UNSUPPORTED on NPU ({total_unsupported} ops - WILL BLOCK VELA):")
            for op, count in sorted(unsupported.items(), key=lambda x: -x[1]):
                print(f"    ✗ {op:30s} x{count}")
        
        if unknown:
            print(f"\n  ? UNKNOWN compatibility ({total_unknown} ops):")
            for op, count in sorted(unknown.items(), key=lambda x: -x[1]):
                print(f"    ? {op:30s} x{count}")
        
        # MAC estimation
        total_macs, conv_details = estimate_mac_ops(model)
        
        if total_macs > 0:
            print(f"\n{'─'*72}")
            print("  COMPUTE ESTIMATION")
            print(f"{'─'*72}")
            print(f"  Total MACs: {total_macs/1e6:.2f} M ({total_macs/1e9:.3f} GOPS)")
            print(f"  Ethos-U55 (0.5 TOPS) estimated: ~{total_macs/500e6*1000:.1f} ms")
            print(f"  Ethos-U55 INT8 speedup:         ~{total_macs/500e6*1000*0.7:.1f} ms (optimistic)")
            if len(conv_details) <= 20:
                print(f"\n  Layer breakdown:")
                for d in conv_details:
                    print(f"  {d}")
        
        # Final verdict
        print(f"\n{'='*72}")
        print("  VERDICT")
        print(f"{'='*72}")
        
        issues = []
        
        if not is_quantized:
            issues.append("NOT QUANTIZED - Model uses FLOAT32, needs INT8 quantization first")
        elif float_count > int_count:
            issues.append("MIXED PRECISION - Has both float and int tensors, may not fully accelerate")
        
        if unsupported:
            ops_list = ", ".join(unsupported.keys())
            issues.append(f"UNSUPPORTED OPS - {ops_list}")
        
        if unknown:
            ops_list = ", ".join(unknown.keys())
            issues.append(f"UNKNOWN OPS - {ops_list} (check Vela docs)")
        
        if not issues:
            print(f"\n  ✓ ✓ ✓  MODEL IS FULLY VELA-COMPATIBLE!  ✓ ✓ ✓")
            print(f"\n  Ready for Vela compilation:")
            print(f"  vela {os.path.basename(filepath)} \\")
            print(f"    --accelerator-config=ethos-u55-256 \\")
            print(f"    --system-config=Ethos_U55_High_End_Embedded \\")
            print(f"    --memory-mode=Shared_Sram \\")
            print(f"    --output-dir=./models_vela")
        else:
            print(f"\n  ✗ ✗ ✗  MODEL HAS COMPATIBILITY ISSUES  ✗ ✗ ✗")
            for i, issue in enumerate(issues, 1):
                print(f"\n  Issue {i}: {issue}")
            
            if unsupported:
                print(f"\n  To fix unsupported ops:")
                print(f"  1. Go back to source model (PyTorch/TF)")
                print(f"  2. Replace unsupported activations:")
                for op in unsupported:
                    if "CUSTOM" in op:
                        print(f"     {op} → Remove or replace with standard op")
                    elif op == "GELU":
                        print(f"     GELU → Replace with RELU6 or LEAKY_RELU")
                    elif op == "ELU":
                        print(f"     ELU → Replace with LEAKY_RELU")
                    else:
                        print(f"     {op} → Check Vela docs for alternative")
                print(f"  3. Re-export to TFLite with INT8 quantization")
                print(f"  4. Compile with Vela")
        
        print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
