import os
import json

# Common parent folder under which every jit_* capture directory is created
# during simulacrum (write) and read back from during reuse. Both phases share
# this root so the captures stay consistent. Relative to CWD by default; the CLI
# overrides it via set_jit_root() from the --jit-dir option.
jit_root = "jit_captures"

def set_jit_root(path):
    global jit_root
    jit_root = path

def jit_path(*parts):
    """Resolve a jit capture path under jit_root.

    Accepts either a bare directory name ("jit_binary") or an already-joined
    relative capture path ("jit_binary/binary_0_1.json"); both are placed under
    the common parent folder.
    """
    return os.path.join(jit_root, *parts)


# When in_memory_captures is set (via `jit --in-memory`), jit captures live in
# this process-local dict instead of on disk: the key is the capture's relative
# path (the same string that would be the on-disk filename, e.g.
# "jit_binary/binary_0_1_False_None_None.json") and the value is the JSON-encoded
# capture, so it is byte-for-byte equivalent to the file it replaces. This only
# works within the single `jit` process.
_jit_store = {}

def jit_store_clear():
    _jit_store.clear()

def save_capture(rel_path, obj):
    """Persist a jit capture under its relative path (dir/file.json)."""
    if in_memory_captures:
        _jit_store[rel_path] = json.dumps(obj)
        return
    path = jit_path(rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)

def load_capture(rel_path):
    """Load a jit capture previously written with save_capture."""
    if in_memory_captures:
        return json.loads(_jit_store[rel_path])
    with open(jit_path(rel_path), 'r') as f:
        return json.load(f)

def capture_exists(rel_path):
    if in_memory_captures:
        return rel_path in _jit_store
    return os.path.exists(jit_path(rel_path))


class Flag:
    def __init__(self):
        self.flag = False

    def set_flag(self):
        self.flag = True

    def reset_flag(self):
        self.flag = False

    def get_flag(self):
        return self.flag
    
    def __bool__(self):
        return self.flag
    

    def __str__(self):
        return f'flag: {self.flag}'

debug_flag = Flag()
debug_flag2 = Flag()
debug_flag3 = Flag()
debug_flag4 = Flag()

dummy_mode = Flag()
reuse_mode = Flag()
dense_default_mode = Flag()
inductor_mode = Flag()
# When set (via --no-barriers), subexp_inlining folds every single-use temporary
# unconditionally: 
no_barriers = Flag()
# When set (via `jit --in-memory`), jit captures are kept in the process-local
# _jit_store dict instead of written to / read from disk. See save_capture.
in_memory_captures = Flag()



class DeviceMode:
    _DEVICE_MAP = {
        'cpu':    'cpu',
        'gpu':    'cuda',
        'gpumac': 'mps',
    }

    def __init__(self):
        self._device = 'cpu'

    def set_cpu(self):
        self._device = 'cpu'

    def set_gpu(self):
        self._device = 'cuda'

    def set_gpumac(self):
        self._device = 'mps'

    def set_mode(self, mode: str):
        if mode not in self._DEVICE_MAP:
            raise ValueError(f"Unknown device mode '{mode}'. Choose from: {list(self._DEVICE_MAP)}")
        self._device = self._DEVICE_MAP[mode]

    def get_device(self) -> str:
        return self._device

    def is_accelerated(self) -> bool:
        return self._device != 'cpu'

    def sync(self):
        if self._device == 'cuda':
            import torch
            torch.cuda.synchronize()

device_mode = DeviceMode()

class OperationTime():
    def __init__(self):
        self.total_time = 0
        self.data_transfer_time = 0
        self.actual_op_time = 0
        self.num_ops = 0
    def update_total_time(self, time1):
        self.total_time += time1
        self.num_ops += 1
    def update_data_transfer_time(self, time1):
        self.data_transfer_time += time1
    def update_actual_op_time(self, time1):
        self.actual_op_time += time1
        
    def get_total_time(self):
        return self.total_time
    def get_data_transfer_time(self):
        return self.data_transfer_time
    def get_actual_op_time(self):        
        return self.actual_op_time
    def get_num_ops(self):
        return self.num_ops


class Time:
    def __init__(self):
        self.total_time = 0
        self.op_time = 0
        self.num_used = 0

    def __str__(self):
        if self.total_time == 0:
            percentage_op_time = 0
        else:
            percentage_op_time = 100*self.op_time/self.total_time 
        return f'total time: {self.total_time: 8.3f}s, \
operation time: {self.op_time: 8.3f}s, \
%op time: {percentage_op_time: 8.3f}, \
index time: {self.total_time - self.op_time: 8.3f}s, \
num used: {self.num_used}'
    
    def just_update_total_time(self, time1):
        self.total_time += time1
    def update_total_time(self, time1):
        self.total_time += time1
        self.num_used += 1

    def update_num_used(self):
        self.num_used += 1

    def update_op_time(self, time1):
        self.op_time += time1

    def get_total_time(self):
        return self.total_time

    def get_op_time(self):
        return self.op_time
    
binary_time = Time()
unary_time = Time()
matmul_time = Time()
where_time = Time()
repeat_time = Time()
clamp_time = Time()
any_time = Time()
all_time = Time()
unsqueeze_time = Time()
get_elem_time = Time()
get_sparse_range_time = Time()
reduce_size_time = Time()
filter_non_live_time = Time()
union_tensors_time = Time()
patches_to_mat_time = Time()
sub_block_custom_range_time = Time()
squeeze_time = Time()
sum_time = Time()
sanity_time = Time()
sparse_tensor_init_time = Time()
binary_profilier = OperationTime()
unary_profilier = OperationTime()
equal_matmul_profilier = OperationTime()
unequal_matmul_profilier = OperationTime()
clamp_profilier = OperationTime()



binary_tensor_ops_expenses = Time()
binary_tensor_ops_x_sparsity = Time()
binary_tensor_ops_y_sparsity = Time()
binary_tensor_ops_no_sparse = Time()
total_binary_tensor_ops = Time()

binary_sparse_tensor_expenses = Time()
binary_sparse_tensor_dom2 = Time()
binary_sparse_tensor_dom1 = Time()
binary_sparse_tensor_overlap = Time()
binary_sparse_tensor_overlap_expenses = Time()
binary_sparse_tensor_dom1_expenses = Time()
binary_sparse_tensor_dom2_expenses = Time()
total_binary_sparse_tensor = Time()

binary_sparse_tensor_count = Time()
binary_block_level_tensor_count = Time()

binary_block_expenses = Time()

binary_fixed_costs = Time()


matmul_tensor_ops = Time()
matmul_tensor_ops_expenses = Time()
matmul_sparse_tensor_expenses = Time()
matmul_block_all= Time()
matmul_block_op = Time()

# Added by Heng
clamp_total_time = Time()
clamp_op_expense = Time()
clamp_sparse_tensor_expense = Time()
clamp_sparse_block_expense = Time()
clamp_sparse_block_op_time = Time()
clamp_const_block_expense = Time()
clamp_const_block_op_time = Time() # Should be 0, added for uniformity.
clamp_repeat_block_expense = Time()
clamp_repeat_block_op_time = Time()

stop_condition_time = Time()
