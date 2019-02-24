"""
This module contains all relevent DRL models for continuous control.
@author: Tianshu Chu
"""

import os
import numpy as np
import tensorflow as tf

import policies
import rl_utils
import train_utils


class DDPG:
    """
    The impelmentation of DDPG (Deep Deterministic Policy Gradient).

    Attributes:
        name (str): DRL algorithm name
        n_a (int): continuous action space dimension
        n_batch (int): minibatch size
        n_update (int): number of updates per MDP step
        n_warmup (int): number of steps doing critic updating only
        lr_scheduler: learning rate scheduler
        policy: DRL policy
        reward_norm (float): reward normalization during gradient computation
        saver: tf saver
        sess: tf session
        total_step (int): max training step
        trans_buffer: experience replay buffer
        v_coef (float): weight of value loss
    """
    def __init__(self, n_s, n_a, total_step=0, model_config=None):
        """
        Initialization.

        Args:
            n_s (int): continuous state space dimension
            n_a (int): continuous action space dimension
            total_step (int): max training step
            model_config: config of hyper-parameters
        """
        self.name = 'ddpg'
        self.reward_norm = model_config.getfloat('reward_norm')
        self.v_coef = model_config.getfloat('value_coef')
        self.n_update = model_config.getint('num_update')
        self.n_warmup = model_config.getfloat('warmup_step')

        # init tf graph
        tf.reset_default_graph()
        tf.set_random_seed(0)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self._init_policy(n_s, n_a, model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        self.n_a = n_a
        if total_step > 0:
            self.total_step = total_step
            self.lr_scheduler = self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def save(self, model_dir, global_step):
        """
        Save the trained model.

        Args:
            model_dir (str): model output path
            global_step (int): current saving step
        """
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        """
        Load the pre-trained model.

        Args:
            model_dir (str): model input path
            checkpoint (int): saving step of model. Load the latest model if it is None.
        """
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            train_utils.print_log('Checkpoint loaded: %s' % save_file)
            return True
        train_utils.print_log('Can not find old checkpoint for %s' % model_dir,
                              level='error')
        return False

    def add_transition(self, ob, action, reward, next_ob, done, ou=True):
        """
        Add current MDP step to experience replay.

        Args:
            ob (array): MDP state
            action (array): MDP action
            reward (float): MDP step reward
            next_ob (array): MDP next state
            done (bool): MDP done flag (if episode is terminated)
        """
        if self.reward_norm:
            reward /= self.reward_norm
        if ou:
            self.trans_buffer.add_transition(ob, action, reward, next_ob, done)
        else:
            self.exp_buffer.add_transition(ob, action, reward, next_ob, done)

    def backward(self, summary_writer=None, global_step=None):
        """
        Backward update on actor and critic.

        Args:
            summary_writer: tf summary writer
            global_step (int): current step
        """
        if self.trans_buffer.size < self.n_batch:
            return
        if self.trans_buffer.size < self.n_warmup:
            warmup = True
        else:
            warmup = False
        cur_lr = self.lr_scheduler.get(1)
        lr_actor, lr_critic = cur_lr, cur_lr * self.v_coef
        # summary: loss_v, loss_p, loss, grad_norm_v, grad_norm_p
        for i in range(self.n_update):
            obs_ou, acts_ou, next_obs_ou, rs_ou, dones_ou = \
                self.trans_buffer.sample_transition()
            obs_exp, acts_exp, next_obs_exp, rs_exp, dones_exp = \
                self.exp_buffer.sample_transition()
            obs = np.vstack([obs_ou, obs_exp])
            acts = np.vstack([acts_ou, acts_exp])
            next_obs = np.vstack([next_obs_ou, next_obs_exp])
            rs = np.concatenate([rs_ou, rs_exp])
            dones = np.concatenate([dones_ou, dones_exp])
            if i == self.n_update-1:
                self.policy.backward(self.sess, obs, acts, next_obs, dones, rs,
                                     lr_critic, lr_actor, warmup=warmup,
                                     summary_writer=summary_writer,
                                     global_step=global_step)
            else:
                self.policy.backward(self.sess, obs, acts, next_obs, dones, rs,
                                     lr_critic, lr_actor, warmup=warmup)

    def forward(self, ob, mode='explore'):
        """
        Get the forward model inference.

        Args:
            ob (array): MDP state
            mode (str): 'explore': random exploration; 'act': greedy policy
        Returns:
            policy (array): DRL policy
        """
        # compeltely random exploration during warmup is disabled
        # if (mode == 'explore') and (self.trans_buffer.size < self.n_warmup):
        #     return np.random.uniform(-1, 1, len(ob))
        return self.policy.forward(self.sess, ob, mode=mode)

    def init_train(self):
        """Model initialization (of target actor and critic DNNs)."""
        self.sess.run(self.policy.init_target)

    def reset_noise(self):
        """Reset exploration OU noise."""
        self.policy.reset_noise()

    def _init_policy(self, n_s, n_a, model_config):
        """
        Initialize DRL policy.

        Args:
            n_s (int): continuous state space dimension
            n_a (int): continuous action space dimension
            model_config: config of hyper-parameters
        """
        # initialize OU noise generator for exploration
        if ('ou_theta' in model_config) and ('ou_sigma' in model_config):
            theta = model_config.getfloat('ou_theta')
            sigma = model_config.getfloat('ou_sigma')
            noise_generator = rl_utils.OUNoise(theta=theta, sigma=sigma)
        else:
            noise_generator = rl_utils.OUNoise()
        n_batch = model_config.getint('batch_size')
        n_fc = model_config.get('num_fc').split(',')
        n_fc = [int(x) for x in n_fc]
        self.n_batch = n_batch
        # initialize DRL policy
        self.policy = policies.DDPGFCPolicy(n_s, n_a, n_batch, n_fc, noise_generator)

    def _init_scheduler(self, model_config, name='lr'):
        """
        Initialize scheduler.

        Args:
            model_config: config of hyper-parameters
            name (str): 'lr': learning rate; 'beta': policy entropy
        Returns:
            scheduler (rl_utils.Scheduler)
        """
        val_init = model_config.getfloat(name + '_init')
        val_decay = model_config.get(name + '_decay')
        if val_decay == 'constant':
            return rl_utils.Scheduler(val_init, decay=val_decay)
        if name + '_min' in model_config:
            val_min = model_config.getfloat(name + '_min')
        else:
            val_min = 0
        decay_step = self.total_step
        if name + '_ratio' in model_config:
            decay_step *= model_config.getfloat(name + '_ratio')
        return rl_utils.Scheduler(val_init, val_min=val_min, total_step=decay_step, decay=val_decay)

    def _init_train(self, model_config):
        """
        Initialize other training components, such as loss function and experience replay buffer.

        Args:
            model_config: config of hyper-parameters
        """
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        tau = model_config.getfloat('tau')
        if 'l2_actor' in model_config:
            l2_actor = model_config.getfloat('l2_actor')
        else:
            l2_actor = 0
        if 'l2_critic' in model_config:
            l2_critic = model_config.getfloat('l2_critic')
        else:
            l2_critic = 0
        # initialize loss function
        self.policy.prepare_loss(self.v_coef, l2_actor, l2_critic, gamma, tau, max_grad_norm)
        buffer_size = model_config.getfloat('buffer_size')
        # initialize experience replay buffer
        self.trans_buffer = rl_utils.ReplayBuffer(buffer_size, int(self.n_batch*7/8))
        self.exp_buffer = rl_utils.ReplayBuffer(buffer_size, int(self.n_batch/8))

