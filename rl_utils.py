"""
This module contains all helper functions for DRL and DNN.
@author: Tianshu Chu
"""

import numpy as np
import random
import tensorflow as tf

DEFAULT_SCALE = np.sqrt(2)
DEFAULT_MODE = 'fan_in'


def ortho_init(scale=DEFAULT_SCALE, mode=None):
    """Orthogonal weight initialization for DNN."""
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            # fc: in, out
            flat_shape = shape
        elif (len(shape) == 3) or (len(shape) == 4):
            # 1d/2dcnn: (in_h), in_w, in_c, out
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (scale * q).astype(np.float32)
    return _ortho_init


def norm_init(scale=DEFAULT_SCALE, mode=DEFAULT_MODE):
    """Standard weight initialization for DNN."""
    def _norm_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            n_in = shape[0]
        elif (len(shape) == 3) or (len(shape) == 4):
            n_in = np.prod(shape[:-1])
        a = np.random.standard_normal(shape)
        if mode == 'fan_in':
            n = n_in
        elif mode == 'fan_out':
            n = shape[-1]
        elif mode == 'fan_avg':
            n = 0.5 * (n_in + shape[-1])
        return (scale * a / np.sqrt(n)).astype(np.float32)

DEFAULT_METHOD = ortho_init

"""
layers
"""
def conv(x, scope, n_out, f_size, stride=1, pad='VALID', f_size_w=None, act=tf.nn.relu,
         conv_dim=1, init_scale=DEFAULT_SCALE, init_mode=None, init_method=DEFAULT_METHOD):
    """Convolutional layer of DNN."""
    with tf.variable_scope(scope):
        b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
        if conv_dim == 1:
            n_c = x.shape[2].value
            w = tf.get_variable("w", [f_size, n_c, n_out],
                                initializer=init_method(init_scale, init_mode))
            z = tf.nn.conv1d(x, w, stride=stride, padding=pad) + b
        elif conv_dim == 2:
            n_c = x.shape[3].value
            if f_size_w is None:
                f_size_w = f_size
            w = tf.get_variable("w", [f_size, f_size_w, n_c, n_out],
                                initializer=init_method(init_scale, init_mode))
            z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad) + b
        return act(z)


def fc(x, scope, n_out, act=tf.nn.relu, init_scale=DEFAULT_SCALE,
       init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    """Fully connected layer of DNN."""
    with tf.variable_scope(scope):
        n_in = x.shape[1].value
        w = tf.get_variable("w", [n_in, n_out],
                            initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        return act(z)


def batch_to_seq(x):
    """Reshape minibatch to LSTM input steps."""
    n_step = x.shape[0].value
    if len(x.shape) == 1:
        x = tf.expand_dims(x, -1)
    return tf.split(axis=0, num_or_size_splits=n_step, value=x)


def seq_to_batch(x):
    """Reshape LSTM output steps to minibatch."""
    return tf.concat(axis=0, values=x)


def lstm(xs, dones, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
         init_method=DEFAULT_METHOD):
    """LSTM layer of DNN."""
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = xs[0].shape[1].value
    n_out = s.shape[0] // 2
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [n_in, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        wh = tf.get_variable("wh", [n_out, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out*4], initializer=tf.constant_initializer(0.0))
    s = tf.expand_dims(s, 0)
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[ind] = h
    s = tf.concat(axis=1, values=[c, h])
    return seq_to_batch(xs), tf.squeeze(s)

"""
experience buffers
"""
class ExpBuffer:
    """Abstract class of experience buffer, has to be overwritten."""
    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()


class OnPolicyBuffer(ExpBuffer):
    """
    On-policy rollout buffer for A2C, A3C training.
    
    Attributes:
        size (int): buffer size
        gamma (float): MDP discount factor
        obs (array): MDP states
        acts (array): MDP actions
        rs (array): MDP rewards
        vs (array): estimated baseline values
        dones (array): MDP done flags
        Rs (array): cost-to-go values (Rs[i] = np.sum(rs[i:]))
        Advs (array): advantages (Advs[i] = Rs[i] - vs[i])
    """
    def __init__(self, gamma):
        """
        Initialization.

        Args:
            gamma (float): MDP discount factor
        """
        self.gamma = gamma
        self.reset()

    def add_transition(self, ob, a, r, v, done):
        """
        Add current MDP step to rollout buffer.

        Args:
            ob (array): MDP state
            a (array): MDP action
            r (float): MDP step reward
            v (float): estimated value
            done (bool): MDP done flag (if episode is terminated)
        """
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def reset(self, done=False):
        """
        Reset rollout buffer.

        Args:
            done (bool): MDP done flag (pre-decision for LSTM reset)
        """
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        # the done before each step is required
        self.dones = [done]

    def sample_transition(self, R, discrete=True):
        """
        Get the current minimatch from rollout buffer.

        Args:
            R (float): terminal value
            discrete (bool): if action is discrete
        Returns:
            obs (array): MDP states
            acts (array): MDP actions
            dones (array): MDP done flags
            Rs (array): cost-to-go values
            Advs (array): advantages
        """
        self._add_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        if discrete:
            acts = np.array(self.acts, dtype=np.int32)
        else:
            acts = np.array(self.acts, dtype=np.float32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, acts, dones, Rs, Advs

    @property
    def size(self):
        return len(self.rs)

    def _add_R_Adv(self, R):
        """Compute cost-to-go values and advantages using DP operator."""
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs


class ReplayBuffer(ExpBuffer):
    """
    Off-policy replay buffer for DDPG, DQN training.
    
    Attributes:
        size (int): buffer size
        buffer (list): buffer list of experiences
        buffer_size (int): max buffer size
        batch_size (int): minibatch size
        cum_size (int): cumulative number of experiences explored
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initialization.

        Args:
            buffer_size (int): max buffer size
            batch_size (int): minibatch size
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cum_size = 0
        self.buffer = []

    def add_transition(self, ob, a, r, next_ob, done):
        """
        Add current MDP step to replay buffer.

        Args:
            ob (array): MDP state
            a (array): MDP action
            r (float): MDP step reward
            next_ob (array): MDP next state
            done (bool): MDP done flag (if episode is terminated)
        """
        experience = (ob, a, r, next_ob, done)
        if self.cum_size < self.buffer_size:
            self.buffer.append(experience)
        else:
            ind = int(self.cum_size % self.buffer_size)
            self.buffer[ind] = experience
        self.cum_size += 1

    def reset(self):
        """Reset replay buffer."""
        self.buffer = []
        self.cum_size = 0

    def sample_transition(self):
        """
        Randomly sample minibatch from replay buffer.

        Returns:
            state_batch (array): MDP states
            action_batch (array): MDP actions
            next_state_batch (array): MDP next states
            reward_batch (array): MDP rewards
            done_batch (array): MDP done flags
        """
        minibatch = random.sample(self.buffer, self.batch_size)
        state_batch = np.array([data[0] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        done_batch = np.array([data[4] for data in minibatch])
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

    @property
    def size(self):
        return min(self.buffer_size, self.cum_size)

"""
util functions
"""
class Scheduler:
    """
    Scheduler for parameters decaying over training steps.

    Attributes:
        decay (str): decay type: 'linear', 'constant'
        N (int): total step to be schedulered over
        n (int): current step
        val (float): current value
        val_min (float): min value
    """
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        """
        Initialization.

        Args:
            val_init (float): initial value
            val_min (float): min value
            total_step (int): total step
            decay (str): decay type: 'linear', 'constant'
        """
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        """
        Advance n steps and get the decayed value.

        Args:
            n_step (int): move forward n steps
        Returns:
            val (float): current value
        """
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


class OUNoise:
    """
    OU noise generator for training DDPG. 
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    Attributes:
        dimension (int): noise dimension
        state (array): current state
        mu, theta, sigma (float): parameters
    """
    def __init__(self, mu=0, theta=0.15, sigma=0.2):
        """
        Initialization.

        Args:
            mu, theta, sigma (float): parameters
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def reset(self, dimension):
        """
        Reset each state to mu.

        Args:
            dimension (int): noise dimension
        """
        self.dimension = dimension
        self.state = np.ones(self.dimension) * self.mu

    def noise(self):
        """
        Perform one step OU noise and sample the next state.

        Returns:
            state (array): sampled state
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.dimension)
        self.state = x + dx
        return self.state

