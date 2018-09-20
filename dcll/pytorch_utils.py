import torch
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2
import gym
import torchvision.utils as vutils

def enforce_1_or_3_channels(im):
    if im.shape[0] == 2:
        pad = torch.zeros(1, im.shape[1], im.shape[2])
        new_im = torch.cat((im.cpu(), pad), dim=0)
        return new_im
    elif im.shape[0] == 0 or im.shape[0] == 3:
        return im
    else:
        raise NotImplementedError

def grad_parameters(module):
    return list(filter(lambda p: p.requires_grad, module.parameters()))

def named_grad_parameters(module):
    return list(filter(lambda p: p[1].requires_grad, module.named_parameters()))

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

class ForwardHook(object):
    def __init__(self, writer, title, initial_time):
        self.writer = writer
        self.title = title
        self.recording_time = initial_time


    def write_data(self, data, comment=""):
        # if dictionary of 1 item, we use the key as comment
        if isinstance(data, dict) and len(data) == 1:
            comment = '_' + str(list(data)[0])
            data = list(data.values())[0]

        # write the data wrt datatype
        if isinstance(data, dict):
            self.writer.add_scalars(self.title + comment, data, self.recording_time)
        elif isinstance(data, torch.Tensor) and len(data.shape) == 1 and data.shape[0] == 1: # single value
            self.writer.add_scalar(self.title + comment, data, self.recording_time)
        elif isinstance(data, torch.Tensor):
            img = vutils.make_grid(data.unsqueeze(dim=1) / torch.max(data)) # grey color channel
            self.writer.add_image(self.title + comment, img, self.recording_time)
        else:
            raise NotImplementedError

    def __call__(self, ctx, input, output):
        layer_debug = output[-1]
        if not isinstance(layer_debug, list):
            self.write_data(layer_debug)
        else:
            [ self.write_data(d, comment='_idx_'+str(i)) for i, d in enumerate(layer_debug) ]
        self.recording_time += 1

class NetworkDumper(object):
    def __init__(self, writer, model):
        self.writer = writer
        self.model = model

    def histogram(self, prefix="", t=0):
        params = self.model.named_parameters()
        for name, param in params:
            self.writer.add_histogram(prefix+name,
                                      param.cpu().detach().numpy().flatten(),
                                      t, bins='fd')
    def weight2d(self, prefix="", t=0):
        params = self.model.named_parameters()
        # filter out all params that don't correspond to convolutions (KCHW)
        params = list(filter(lambda p: len(p[1].shape) == 4 and
                             ((p[1].shape[1] == 1) or
                              (p[1].shape[1] == 2) or
                              (p[1].shape[1] == 3)), params))
        def enforce_1_or_3_channels(p):
            if not p[1].shape[1] == 2:
                return p
            else:
                pad = torch.zeros(p[1].shape[0], 1, p[1].shape[2], p[1].shape[3])
                new_p = ( p[0],
                          torch.cat((p[1].cpu(), pad), dim=1) )
                return new_p

        params = list(map(enforce_1_or_3_channels, params))
        all_filters = [
            vutils.make_grid(l[1]).unsqueeze(dim=1)
            for l in params
        ]
        for i, img in enumerate(all_filters):
            self.writer.add_image(prefix+params[i][0],
                                  img, t)

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

    def start_recording(self, title="forward_data", t=0):
        hook = ForwardHook(self.writer, title, t)
        # register and return the handle
        return self.model.register_forward_hook(hook)


"""Transform a gym environment's observation space to pixel through the render() method"""
class FrameObs(gym.ObservationWrapper):
    def __init__(self, env, width=32, height=32, crop=True, render=False):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, high=1.,
                                                # torch order (CHW)
                                                shape=(1, self.height, self.width))
        self.render_plot = render
        self.crop = crop
        if (self.render_plot):
            self.obs_plot = plt.imshow(np.zeros((self.height, self.width)))

    def observation(self, obs):
        frame = self.render(mode='rgb_array')
        return frame

    def _observation(self, obs):
        return self.observation(obs)

    def render(self, mode, **kwargs):
        frame = self.env.render(mode, **kwargs)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self.crop:
            half = np.array(frame.shape) // 2
            frame = frame[
                half[0]-self.height // 2 : half[0]+self.height // 2,
                half[1]-self.width // 2 : half[1]+self.width // 2
            ]
        else:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        frame = 1. - np.ascontiguousarray(frame, dtype=np.float32) / 255.
        if (self.render_plot):
            self.obs_plot.set_data(frame)
            plt.imshow(frame, cmap=plt.get_cmap('gray'))
            plt.pause(0.0001)
        frame = np.expand_dims(frame, axis=0) # unsqueeze channel into torch order (CHW)
        return frame

class DVSObs(FrameObs):
    def __init__(self, env, width=32, height=32, crop=True, render=False, threshold=0.1):
        super(DVSObs, self).__init__(env, width, height, crop, render)
        # DVS has two channels: ON/OFF
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                # torch is (CHW)
                                                shape=(2, self.height, self.width))
        self.current_frame = np.zeros((self.height, self.width))

        if (self.render_plot):
            self.obs_plot = plt.imshow(np.zeros((self.height, self.width, 3)))

        self.threshold = threshold

    def render(self, mode, **kwargs):
        frame = self.env.render(mode, **kwargs)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self.crop:
            half = np.array(frame.shape) // 2
            frame = frame[
                half[0]-self.height // 2 : half[0]+self.height // 2,
                half[1]-self.width // 2 : half[1]+self.width // 2
            ]
        else:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        frame = np.ascontiguousarray(frame, dtype=np.float32) / 255.
        diff = frame - self.current_frame
        # zero out all pixels not above threshold
        diff[(-self.threshold < diff) & (diff < self.threshold)] = 0.
        # split positive and negative channels
        ret = np.array( [ diff > 0, diff < 0 ], dtype=np.float32)
        # update the current frame
        events = np.nonzero(diff)
        self.current_frame[events] += diff[events]

        if (self.render_plot):
            self.obs_plot.set_data(frame)
            rgb = np.append(ret, [np.zeros_like(ret[0])], axis=0)
            plt.imshow(np.moveaxis(rgb, 0, 2))
            plt.pause(0.0001)
        return ret

class AERObs(gym.ObservationWrapper):
    def __init__(self, env, threshold=0.1):
        gym.ObservationWrapper.__init__(self, env)
        # DVS has two channels: ON/OFF
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                # torch is (CHW)
                                                shape=(2,
                                                       env.observation_space.shape[0],
                                                       env.observation_space.shape[1]))
        self.normalizer = env.observation_space.high
        self.current_frame = np.zeros(env.observation_space.shape)
        self.threshold = threshold

    def observation(self, obs):
        if len(obs.shape) > 2:
            obs = cv2.cvtColor(obs / self.normalizer, cv2.COLOR_RGB2GRAY)
        frame = np.ascontiguousarray(obs, dtype=np.float32)
        diff = frame - self.current_frame
        # zero out all pixels not above threshold
        diff[(-self.threshold < diff) & (diff < self.threshold)] = 0.
        # split positive and negative channels
        ret = np.array( [ diff > 0, diff < 0 ], dtype=np.float32)
        # update the current frame
        events = np.nonzero(diff)
        self.current_frame[events] += diff[events]
        self.current_events = ret
        return ret

    def _observation(self, obs):
        return self.observation(obs)

    def render(self, mode='human', **kwargs):
        rgb = np.append(self.current_events, [np.zeros_like(self.current_events[0])], axis=0)
        rgb = np.moveaxis(rgb, 0, 2)

        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, self.ax_full = plt.subplots()
        self.ax_full.axis('off')

        self.fig.show()
        # Only create the image the first time
        if not hasattr(self, 'ax_full_img'):
            self.ax_full_img = self.ax_full.imshow(rgb, animated=True)
        # Update the image data for efficient live video
        self.ax_full_img.set_data(rgb)
        plt.draw()
        # Update the figure display immediately
        self.fig.canvas.draw()

        return self.fig

class FrameResizer(gym.ObservationWrapper):
    def __init__(self, env, shape):
        gym.ObservationWrapper.__init__(self, env)
        if len(env.observation_space.shape) > 2:
            # torch is (CHW)
            shape = (env.observation_space.shape, shape[0], shape[1])
        else:
            shape = (shape[0], shape[1])

        self.observation_space = gym.spaces.Box(low=env.observation_space.low[0][0],
                                                high=env.observation_space.high[0][0],
                                                shape=shape)

        self.shape = shape
        self.last_observation = np.zeros(shape)

    def observation(self, obs):
        self.last_observation = cv2.resize(obs, self.shape)
        return self.last_observation

    def _observation(self, obs):
        return self.observation(obs)

    def render(self, mode='human', **kwargs):
        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, self.ax_full = plt.subplots()
            if hasattr(self.env.unwrapped, 'title'):
                self.ax_full.set_title(self.env.unwrapped.title)
        self.ax_full.axis('off')

        self.fig.show()
        # Only create the image the first time
        if not hasattr(self, 'ax_full_img'):
            self.ax_full_img = self.ax_full.imshow(self.last_observation, animated=True, cmap='viridis',
                                                   vmin = self.observation_space.low[0][0],
                                                   vmax = self.observation_space.high[0][0]
            )
        # Update the image data for efficient live video
        self.ax_full_img.set_data(self.last_observation)
        plt.draw()
        # Update the figure display immediately
        self.fig.canvas.draw()

        return self.fig
