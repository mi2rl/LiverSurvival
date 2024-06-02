from torch.nn import BatchNorm3d, InstanceNorm3d, GroupNorm, SyncBatchNorm

def _normalization_3d(inputs, norm='bn'):
    if norm == 'bn':
        return BatchNorm3d(inputs)
        #return SyncBatchNorm(inputs)
    elif norm == 'in':
        return InstanceNorm3d(inputs)
    elif norm == 'gn':
        #return GroupNorm(max(32, inputs), inputs)
        return GroupNorm(32, inputs)