import torch
import math
import onnx
import numpy as np
import torch.nn as nn
import copy
import os

from onnx import numpy_helper
from constraintflow.lib.globals import device_mode
from constraintflow.lib.network import Layer, LayerType, Network

from collections import deque

# This would be cheating hence I have removed this. 
# _cuda_initializer_cache_key = None
# _cuda_initializer_cache = None
# def _initializer_tensors(net, net_name=None):
#     """Materialize ONNX initializers, caching one network on the active CUDA device."""
#     global _cuda_initializer_cache_key, _cuda_initializer_cache

#     if device_mode.get_device() != 'cuda' or net_name is None:
#         return {
#             init_vals.name: torch.tensor(numpy_helper.to_array(init_vals))
#             for init_vals in net.graph.initializer
#         }

#     stat = os.stat(net_name)
#     cache_key = (
#         os.path.realpath(net_name),
#         stat.st_mtime_ns,
#         stat.st_size,
#         torch.cuda.current_device(),
#     )
#     if cache_key != _cuda_initializer_cache_key:
#         _cuda_initializer_cache = {
#             init_vals.name: torch.tensor(
#                 numpy_helper.to_array(init_vals),
#                 device=device_mode.get_device(),
#             )
#             for init_vals in net.graph.initializer
#         }
#         _cuda_initializer_cache_key = cache_key
#     return _cuda_initializer_cache

def _initializer_tensors(net, net_name=None):
    target_device = (
        device_mode.get_device()
        if device_mode.get_device() == "cuda" and net_name is not None
        else "cpu"
    )
    return {
        init_vals.name: torch.tensor(
            numpy_helper.to_array(init_vals),
            device=target_device,
        )
        for init_vals in net.graph.initializer
    }
def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s


def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'torch'
    if 'onnx' in net_name:
        net_format = 'onnx'
    return net_format

def get_net(net_name, spec_weight, spec_bias, no_sparsity):
    net_format = get_net_format(net_name)
    if net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        # net type: constraintflow.lib.network.Network (inherits list)
        # net element type: constraintflow.lib.network.Layer
        model_name_to_val_dict = _initializer_tensors(net_onnx, net_name)
        net = parse_onnx_layers(
            net_onnx,
            spec_weight,
            spec_bias,
            no_sparsity,
            model_name_to_val_dict=model_name_to_val_dict,
        )
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name

    print('Network loaded')
    return net

def forward_layers(net, relu_mask, transformers):
    for layer in net:
        if layer.type == LayerType.ReLU:
            transformers.handle_relu(layer, optimize=True, relu_mask=relu_mask)
        elif layer.type == LayerType.Linear:
            if layer == net[-1]:
                transformers.handle_linear(layer, last_layer=True)
            else:
                transformers.handle_linear(layer)
        elif layer.type == LayerType.Conv2D:
            transformers.handle_conv2d(layer)
        elif layer.type == LayerType.Normalization:
            transformers.handle_normalization(layer)
    return transformers


def _shape_value(node, values, shapes):
    op = node.op_type
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if op == 'Constant':
        if 'value' in attrs:
            return numpy_helper.to_array(attrs['value'])
        for key in ('value_int', 'value_ints', 'value_float', 'value_floats'):
            if key in attrs:
                return np.asarray(attrs[key])
        raise ValueError(f"Unsupported Constant {node.name!r}")
    if op == 'Shape':
        shape = shapes.get(node.input[0])
        if shape is None:
            raise ValueError(f"Shape {node.name!r} has an unresolved input")
        return np.asarray(shape[attrs.get('start', 0):attrs.get('end')], dtype=np.int64)
    if op not in ('Gather', 'Unsqueeze', 'Squeeze', 'Concat', 'Mul', 'Add', 'Sub', 'Div', 'Cast'):
        return None
    if not all(name in values for name in node.input):
        if op in ('Gather', 'Unsqueeze', 'Squeeze', 'Cast'):
            raise ValueError(f"Unsupported activation operation {op} {node.name!r}")
        return None
    args = [values[name] for name in node.input]
    if op == 'Gather':
        return np.take(args[0], args[1], axis=attrs.get('axis', 0))
    if op in ('Unsqueeze', 'Squeeze'):
        axes = args[1].tolist() if len(args) > 1 else attrs.get('axes')
        axes = tuple(axes) if axes is not None else None
        return np.expand_dims(args[0], axes) if op == 'Unsqueeze' else np.squeeze(args[0], axes)
    if op == 'Concat':
        return np.concatenate(args, axis=attrs['axis'])
    if op == 'Cast':
        return args[0].astype(onnx.helper.tensor_dtype_to_np_dtype(attrs['to']))
    if op == 'Div':
        result = np.divide(*args)
        return np.trunc(result).astype(args[0].dtype) if np.issubdtype(args[0].dtype, np.integer) else result
    return {'Mul': np.multiply, 'Add': np.add, 'Sub': np.subtract}[op](*args)


def _reshape_shape(node, source, values):
    target = values.get(node.input[1])
    if target is None or target.ndim != 1 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError(f"Reshape {node.name!r} requires a resolved integer shape")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    shape = target.tolist()
    if not attrs.get('allowzero', 0):
        for i, dim in enumerate(shape):
            if dim == 0:
                if i >= len(source):
                    raise ValueError(f"Reshape {node.name!r} has an invalid zero dimension")
                shape[i] = source[i]
    size = math.prod(source)
    if shape.count(-1) > 1 or any(dim < -1 or dim == 0 for dim in shape):
        raise ValueError(f"Reshape {node.name!r} has invalid dimensions {shape}")
    if -1 in shape:
        known = -math.prod(shape)
        if size % known:
            raise ValueError(f"Reshape {node.name!r} changes the neuron count")
        shape[shape.index(-1)] = size // known
    if math.prod(shape) != size or len(shape) not in (2, 4) or shape[0] != 1:
        raise ValueError(f"Unsupported Reshape {node.name!r}: {source} -> {shape}")
    return shape


def parse_onnx_layers(
    net,
    spec_weight,
    spec_bias,
    no_sparsity,
    model_name_to_val_dict=None,
):
    input_shape = [dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim]
    input_shape = [1 if i == 0 else i for i in input_shape]
    if len(input_shape)==3:
        input_shape = [1] + input_shape
    input_size = compute_size(input_shape)
    # Create the new Network object
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape, input_size=input_size, input_start=0, input_end=input_size, net_format='onnx', no_sparsity=no_sparsity)
    num_layers = len(net.graph.node)
    layers.num_layers = num_layers
    if model_name_to_val_dict is None:
        model_name_to_val_dict = _initializer_tensors(net)

    values = {v.name: numpy_helper.to_array(v) for v in net.graph.initializer}
    tensor_shapes = {net.graph.input[0].name: input_shape}
    layers.size = input_size
    shape = input_shape

    names_hash = dict()
    index = 0
    names_hash[net.graph.input[0].name] = index
    parents = dict()
    
    layer = Layer(type=LayerType.Input, shape=input_shape, size=input_size, start=0, end=input_size)
    layers.append(layer)
    for cur_layer in range(num_layers):
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input
        value = _shape_value(node, values, tensor_shapes)
        if value is not None:
            values[node.output[0]] = value
            tensor_shapes[node.output[0]] = list(value.shape)
            continue
        if operation in ('Reshape', 'Flatten', 'Identity'):
            source = tensor_shapes[nd_inps[0]]
            result_shape = source
            if operation == 'Reshape':
                result_shape = _reshape_shape(node, source, values)
            elif operation == 'Flatten':
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                axis = attrs.get('axis', 1)
                if axis < 0:
                    axis += len(source)
                if axis != 1:
                    raise ValueError(f"Unsupported Flatten axis {axis}")
                result_shape = [source[0], math.prod(source[1:])]
            names_hash[node.output[0]] = names_hash[nd_inps[0]]
            tensor_shapes[node.output[0]] = result_shape
            continue
        index+=1

        if operation == 'Conv':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]

            if isinstance(nd_inps[0], str):
                w_key = None
                b_key = None 
                for i in range(len(nd_inps)):
                    if 'weight' in nd_inps[i]:
                        w_key = nd_inps[i]
                    if 'bias' in nd_inps[i]:
                        b_key = nd_inps[i]
                if b_key == None:
                    weight = model_name_to_val_dict[w_key]
                    bias = torch.zeros(
                        weight.shape[0],
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    layer = Layer(weight=weight, bias=bias, type=LayerType.Conv2D, identifier=index, parents=parents[index])
                else:
                    layer = Layer(weight=model_name_to_val_dict[w_key], bias=model_name_to_val_dict[b_key], type=LayerType.Conv2D, identifier=index, parents=parents[index])

            else:
                layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Conv2D, identifier=index, parents=parents[index])
            layers.append(layer)

            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            
            shape = tensor_shapes[nd_inps[0]]
            if shape != layers[parents[index][0]].shape:
                raise ValueError(f"Conv {node.name!r} cannot consume a spatially reshaped alias")
            [i_1, i_2, i_3, i_4] = shape
            [k_1, k_2, k_3, k_4] = layer.weight.shape 
            (p_1, p_2) = layer.padding
            (s_1, s_2) = layer.stride 
            
            o_1 = i_1 
            o_2 = k_1 
            o_3 = math.floor((i_3 + 2*p_1 - k_3) / s_1) + 1
            o_4 = math.floor((i_4 + 2*p_2 - k_4) / s_2) + 1
            layer.shape = [o_1, o_2, o_3, o_4]
            layer.bias = (layer.bias.unsqueeze(1).repeat(1, o_3*o_4)).flatten()

            

        elif operation == 'Gemm':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]), type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            [i_1, i_2, i_3, i_4] = shape 
            [w_1, w_2] = layer.weight.shape 
            o_1 = i_1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]

            
        elif operation == 'Relu':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            layer = Layer(type=LayerType.ReLU, identifier=index, parents=parents[index])
            layers.append(layer)
            source_shape = tensor_shapes[nd_inps[0]]
            layer.shape = source_shape + [1] * (4 - len(source_shape))

        elif operation == 'Sigmoid':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            layer = Layer(type=LayerType.Sigmoid, identifier=index, parents=parents[index])
            layers.append(layer)
            source_shape = tensor_shapes[nd_inps[0]]
            layer.shape = source_shape + [1] * (4 - len(source_shape))

            


        elif operation == 'Concat':
            if len(nd_inps) != 2 or not all(name in names_hash for name in nd_inps):
                raise ValueError(f"Concat {node.name!r} requires two activation inputs")
            left, right = (tensor_shapes[name] for name in nd_inps)
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            axis = attrs['axis']
            if axis < 0:
                axis += len(left)
            if (len(left) != len(right) or not 0 < axis < len(left)
                    or math.prod(left[:axis]) != 1
                    or any(a != b for i, (a, b) in enumerate(zip(left, right)) if i != axis)):
                raise ValueError(f"Unsupported Concat {node.name!r}: axis {axis}, {left}, {right}")
            result_shape = list(left)
            result_shape[axis] += right[axis]
            names_hash[node.output[0]] = index
            parents[index] = [names_hash[name] for name in nd_inps]
            layer = Layer(type=LayerType.Concat, identifier=index, parents=parents[index])
            layers.append(layer)
            layer.shape = result_shape + [1] * (4 - len(result_shape))

        elif operation == 'MatMul':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            constant_inputs = [name in model_name_to_val_dict for name in nd_inps]
            if sum(constant_inputs) != 1:
                raise ValueError(
                    f"MatMul {node.name!r} requires exactly one initializer operand"
                )
            weight_index = constant_inputs.index(True)
            weight = model_name_to_val_dict[nd_inps[weight_index]]
            if weight.ndim != 2:
                raise ValueError(
                    f"MatMul {node.name!r} requires a 2D weight, got {tuple(weight.shape)}"
                )
            # Linear layers store [out_features, in_features]. For X @ W,
            # ONNX stores W as [in_features, out_features].
            if weight_index == 1:
                weight = weight.T.contiguous()
            parents[index] = [names_hash[nd_inps[1 - weight_index]]]
            layer = Layer(weight=weight, type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            [w_1, w_2] = layer.weight.shape 
            o_1 = 1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]

        
        elif operation == 'Add':
            if nd_inps[1] not in model_name_to_val_dict:
                names_hash[str(net.graph.node[cur_layer].output[0])] = index
                parent1 = names_hash[nd_inps[0]]
                parent2 = names_hash[nd_inps[1]]
                parents[index] = [parent1, parent2]
                layer = Layer(weight=None, type=LayerType.Add, identifier=index, parents=parents[index])
                layers.append(layer)
                layer.shape = copy.deepcopy(layers[parents[index][0]].shape)
            else:
                index -= 1
                names_hash[str(net.graph.node[cur_layer].output[0])] = index
                layer = layers[-1]
                layer.bias = model_name_to_val_dict[nd_inps[1]]
                tensor_shapes[node.output[0]] = tensor_shapes[nd_inps[0]]
                continue


            


        else:
            # Some layers are skipped and regarded as identity?
            # How can soundness still hold then?
            if len(net.graph.node[cur_layer].input)>0:
                if str(net.graph.node[cur_layer].input[0]) in names_hash:
                    names_hash[str(net.graph.node[cur_layer].output[0])] = names_hash[str(net.graph.node[cur_layer].input[0])]
                    tensor_shapes[node.output[0]] = tensor_shapes[nd_inps[0]]

            index-=1
            # assert(f"{operation} not supported")
            continue

        if operation in ('MatMul', 'Gemm'):
            tensor_shapes[node.output[0]] = [1, layer.shape[1]]
        elif operation in ('Relu', 'Sigmoid', 'Add'):
            tensor_shapes[node.output[0]] = tensor_shapes[nd_inps[0]]
        elif operation == 'Concat':
            tensor_shapes[node.output[0]] = result_shape
        else:
            tensor_shapes[node.output[0]] = list(layer.shape)
        layer.size = compute_size(layer.shape)
        layer.start = layers.size 
        layers.size += layer.size 
        layer.end = layers.size
        shape = layer.shape

    if True or layers[-1].type != LayerType.Linear:
        layer = Layer(weight=spec_weight, bias=spec_bias, type=LayerType.Linear, identifier=index+1, parents=[layers[-1].identifier])
        layer.last_layer = True
        layer.shape = [1, spec_weight.shape[-2], 1, 1]
        layer.size = compute_size(layer.shape)
        layer.start = layers.size 
        layers.size += layer.size 
        layer.end = layers.size
        layers.append(layer)
    else:
        layers[-1].weight = (spec_weight@layers[-1].weight)[0] 
        layers[-1].bias = (spec_weight@layers[-1].bias + spec_bias)[0]
        layers[-1].shape[1] = spec_weight.shape[-2]
        layers.size = layers.size - layers[-1].size
        layers[-1].size = compute_size(layers[-1].shape)
        layers.size += layers[-1].size
        layers[-1].end = layers.size
    layers = post_process(layers)
    return layers




def post_process(layers):
    identifier_to_index = dict()
    for i, layer in enumerate(layers):
        identifier_to_index[layer.identifier] = i
    for i, layer in enumerate(layers):
        for j, parent in enumerate(layer.parents):
            layers[identifier_to_index[parent]].children.append(layer.identifier)
    queue = deque([layers[identifier_to_index[0]]])
    visited = set()
    layer_num = 0
    new_layers = layers.similar()
    while queue:
        layer = queue.popleft()
        if layer in visited:
            continue
        layer.new_identifier = layer_num
        new_layers.append(layer)
        layer_num += 1
        visited.add(layer)
        for child in layer.children:
            if child not in visited:
                parent_visited = True 
                for parent in layers[identifier_to_index[child]].parents:
                    if layers[identifier_to_index[parent]] not in visited:
                        parent_visited = False
                        break
                if parent_visited:
                    queue.append(layers[identifier_to_index[child]])
    
    start = 0
    for i, layer in enumerate(new_layers):
        layer.start = start 
        layer.end = start + layer.size
        parents = []
        for parent in layer.parents:
            parents.append(layers[parent].new_identifier)
        layer.new_parents = parents
        children = []
        for child in layer.children:
            children.append(layers[child].new_identifier)
        layer.new_children = children
        start += layer.size
    for i,layer in enumerate(new_layers):
        layer.identifier = layer.new_identifier
        layer.parents = layer.new_parents
        layer.children = layer.new_children

    # feeds_nonlin: False iff every immediate consumer is itself Affine
    # (Linear/Conv2D), i.e. this layer's l/u are never read by a relaxation.
    # A layer with no children (only the synthesized spec layer) is conservatively
    # left at the True default set in Layer.__init__.
    for layer in new_layers:
        if layer.children:
            layer.feeds_nonlin = not all(
                new_layers[child].type in (LayerType.Linear, LayerType.Conv2D)
                for child in layer.children
            )

    return new_layers


def parse_torch_layers(net, input_shape):
    if len(input_shape)==3:
        input_shape = [1] + input_shape
    input_size = compute_size(input_shape)

    # Create the new Network object
    layers = Network(input_name='input.torch', input_shape=input_shape, input_size=input_size, input_start=0, input_end=input_size, net_format='torch')
    layers.num_layers = 1

    layers.size = input_size
    shape = input_shape

    names_hash = dict()
    index = 0
    names_hash[net.graph.input[0].name] = index
    parents = dict()
    
    layer = Layer(type=LayerType.Input, shape=input_shape, size=input_size, start=0, end=input_size)
    layers.append(layer)

    for cur_layer, torch_layer in enumerate(net.blocks):
        index+=1

        if isinstance(torch_layer, torch.nn.Conv2d):
            names_hash[cur_layer] = index
            parents[index] = [names_hash[cur_layer-1]]

            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Conv2D, identifier=index, parents=parents[index])
            layers.append(layer)

            layer.kernel_size = torch_layer.kernel_size
            layer.padding = (torch_layer.padding, torch_layer.padding)
            layer.stride = (torch_layer.stride, torch_layer.stride)
            layer.dilation = (torch_layer.dilation, torch_layer.dilation)
            
            shape = layers[parents[index][0]].shape
            [i_1, i_2, i_3, i_4] = shape
            [k_1, k_2, k_3, k_4] = layer.weight.shape 
            (p_1, p_2) = layer.padding
            (s_1, s_2) = layer.stride 
            
            o_1 = i_1 
            o_2 = k_1 
            o_3 = math.floor((i_3 + 2*p_1 - k_3) / s_1) + 1
            o_4 = math.floor((i_4 + 2*p_2 - k_4) / s_2) + 1
            layer.shape = [o_1, o_2, o_3, o_4]
            layer.bias = (layer.bias.unsqueeze(1).repeat(1, o_3*o_4)).flatten()


        elif isinstance(torch_layer, torch.nn.ReLU):
            names_hash[cur_layer] = index
            parents[index] = [names_hash[cur_layer-1]]
            
            layer = Layer(type=LayerType.ReLU, identifier=index, parents=parents[index])
            layers.append(layer)
            layer.shape = layers[parents[index][0]].shape

        elif isinstance(torch_layer, torch.nn.Linear):
            names_hash[cur_layer] = index
            parents[index] = [names_hash[cur_layer-1]]
            
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            [i_1, i_2, i_3, i_4] = shape 
            [w_1, w_2] = layer.weight.shape 
            o_1 = i_1 
            o_2 = w_1 
            o_3 = 1 
            o_4 = 1 
            layer.shape = [o_1, o_2, o_3, o_4]

            
            
        else:
            # print(type(torch_layer))
            assert(False)
    return layers


def load_pytorch_network(dataset, n_class=10, input_size=32, input_channel=3, conv_widths=None,
                 kernel_sizes=None, linear_sizes=None, depth_conv=None, paddings=None, strides=None,
                 dilations=None, pool=False, net_dim=None, bn=False, bn2=False, max=False, scale_width=True, mean=0, sigma=1):
    if kernel_sizes is None:
        kernel_sizes = [3]
    if conv_widths is None:
        conv_widths = [2]
    if linear_sizes is None:
        linear_sizes = [200]
    if paddings is None:
        paddings = [1]
    if strides is None:
        strides = [2]
    if dilations is None:
        dilations = [1]
    if net_dim is None:
        net_dim = input_size

    if len(conv_widths) != len(kernel_sizes):
        kernel_sizes = len(conv_widths) * [kernel_sizes[0]]
    if len(conv_widths) != len(paddings):
        paddings = len(conv_widths) * [paddings[0]]
    if len(conv_widths) != len(strides):
        strides = len(conv_widths) * [strides[0]]
    if len(conv_widths) != len(dilations):
        dilations = len(conv_widths) * [dilations[0]]

    if dataset == "fashionmnist":
        mean = 0.1307
        sigma = 0.3081
    elif dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        sigma = [0.2023, 0.1994, 0.2010]
    elif dataset == "tinyimagenet":
        mean = [0.4802, 0.4481, 0.3975]
        sigma = [0.2302, 0.2265, 0.2262]

    layers = []
    # layers += [Normalization((input_channel,input_size,input_size),mean, sigma)]

    N = net_dim
    n_channels = input_channel
    dims = [(n_channels,N,N)]

    for width, kernel_size, padding, stride, dilation in zip(conv_widths, kernel_sizes, paddings, strides, dilations):
        if scale_width:
            width *= 16
        N = int(np.floor((N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
        layers += [nn.Conv2d(n_channels, int(width), kernel_size, stride=stride, padding=padding, dilation=dilation)]
        if bn:
            layers += [nn.BatchNorm2d(int(width))]
        if max:
            layers += [nn.MaxPool2d(int(width))]
        layers += [nn.ReLU((int(width), N, N))]
        n_channels = int(width)
        dims += 2*[(n_channels,N,N)]

    if depth_conv is not None:
        layers += [nn.Conv2d(n_channels, depth_conv, 1, stride=1, padding=0),
                    nn.ReLU((n_channels, N, N))]
        n_channels = depth_conv
        dims += 2*[(n_channels,N,N)]

    if pool:
        layers += [nn.GlobalAvgPool2d()]
        dims += 2 * [(n_channels, 1, 1)]
        N=1

    layers += [nn.Flatten()]
    N = n_channels * N ** 2
    dims += [(N,)]

    for width in linear_sizes:
        if width == 0:
            continue
        layers += [nn.Linear(int(N), int(width))]
        if bn2:
            layers += [nn.BatchNorm1d(int(width))]
        layers += [nn.ReLU(width)]
        N = width
        dims+=2*[(N,)]

    layers += [nn.Linear(N, n_class)]
    dims+=[(n_class,)]

    blocks = nn.Sequential(*layers)

    return blocks
