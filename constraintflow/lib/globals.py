class Flag:
    def __init__(self):
        self.flag = False

    def set_flag(self):
        self.flag = True

    def reset_flag(self):
        self.flag = False

    def get_flag(self):
        return self.flag
    
    def __str__(self):
        return f'flag: {self.flag}'

debug_flag = Flag()
debug_flag2 = Flag()
debug_flag3 = Flag()
debug_flag4 = Flag()

dummy_mode = False

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
