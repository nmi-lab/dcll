import torch
import numpy as np

def grad_parameters(module):
    return filter(lambda p: p.requires_grad, module.parameters())

def named_grad_parameters(module):
    return filter(lambda p: p[1].requires_grad, module.named_parameters())

def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

class NetworkDumper(object):
    def __init__(self, writer, model):
        self.writer = writer
        self.model = model
        self.handle = None

    def histogram(self, prefix="", t=0):
        params = named_grad_parameters(self.model)
        for name, param in params:
            self.writer.add_histogram(prefix+name,
                                      param.cpu().detach().numpy().flatten(),
                                      t, bins='fd')
    # cache the current parameters of the model
    def cache(self):
        self.cached = named_grad_parameters(self.model)

    # plot the delta histogram between current and cached parameters (weight updates)
    def diff_histogram(self, prefix="", t=0):
        params = named_grad_parameters(self.model)
        for i, (name, param) in enumerate(params):
            self.writer.add_histogram(prefix+'delta_'+name,
                                      param.cpu().detach().numpy().flatten() - self.cached[i],
                                      t, bins='fd')

    def start_recording(self, what_to_record, title="forward_data", t=0):
        hook = lambda *args, **kw: self.forward_hook(what_to_record, *args, **kw)
        self.handle = self.model.register_forward_hook(hook)
        self.recording_time = t
        self.data_name = title

    def forward_hook(self, what_to_record, *args):
        # import ipdb; ipdb.set_trace()
        data = what_to_record(*args)
        if isinstance(data, dict):
            self.writer.add_scalars(self.data_name, data, self.recording_time)
        else:
            self.writer.add_scalar(self.data_name, data, self.recording_time)

        self.recording_time += 1

    def stop_recording(self):
        self.handle.remove()
