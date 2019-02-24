"""
This module contains all relevent DRL policies for continuous control.
@author: Tianshu Chu
"""

import numpy as np
import tensorflow as tf

import rl_utils


class DDPGPolicy:
    """
    The implementation of the general centralized DDPG policy. 
    A subclass is needed to obtain the actual policy using a particular DNN.

    Attributes:
        n_a (int): continuous action space dimension
        n_s (int): continuous state space dimension
        n_step (int): minibatch size
        noise_generator (rl_utils.OUNoise): OU noise generator
        lr_v (tf placeholder): learning rate of critic
        lr_p (tf placeholder): learning rate of actor
        summary (tf summary): summary for tensorboard visualization
    """
    def __init__(self, n_a, n_s, n_step, noise_generator):
        """Poilcy initialization."""
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.noise_generator = noise_generator

    def reset_noise(self):
        """Reset OU noise for exploration."""
        self.noise_generator.reset(self.n_a)

    def _build_fc_net(self, h, n_fc, out_type):
        # TODO: add batch_norm layer before activation
        for i, n_fc_cur in enumerate(n_fc):
            fc_cur = 'fc%d' % i
            h = rl_utils.fc(h, fc_cur, n_fc_cur)
        if out_type == 'pi':
            mu = rl_utils.fc(h, 'out', self.n_a, act=tf.nn.tanh)
            return mu
        else:
            v = rl_utils.fc(h, 'out', 1, act=lambda x: x)
            return tf.squeeze(v, axis=1)

    @staticmethod
    def _init_target_update(cur_vars, tar_vars, tau):
        soft_updates = []
        init_updates = []
        for var, tar_var in zip(cur_vars, tar_vars):
            init_updates.append(tar_var.assign(var))
            soft_updates.append(tar_var.assign_sub(tau * (tar_var - var)))
        return tf.group(*init_updates), tf.group(*soft_updates)

    def prepare_loss(self, v_coef, l2_p, l2_v, gamma, tau, max_grad_norm):
        """Define the loss function and key tf operations for forward inference and 
        backward update.
        """
        # only global policy is available
        # critic net update
        tq = tf.stop_gradient(tf.where(self.DONE, self.R, self.R + gamma * self.q1))
        vars_v = tf.trainable_variables(scope=self.name + '_critic')
        wts_v = [var for var in vars_v if (var.name.endswith('w:0') and
                 'out' not in var.name)]
        loss_v = tf.reduce_mean(tf.square(self.q0 - tq)) + \
            tf.add_n([l2_v * tf.nn.l2_loss(wt) for wt in wts_v])
        grads_v = tf.gradients(loss_v, vars_v)
        if max_grad_norm > 0:
            grads_v, grad_norm_v = tf.clip_by_global_norm(grads_v, max_grad_norm)
        self.lr_v = tf.placeholder(tf.float32, [])
        optimizer_v = tf.train.AdamOptimizer(learning_rate=self.lr_v)
        self._train_v = optimizer_v.apply_gradients(list(zip(grads_v, vars_v)))

        # actor net upadte
        vars_p = tf.trainable_variables(scope=self.name + '_actor')
        wts_p = [var for var in vars_p if (var.name.endswith('w:0') and
                 'out' not in var.name)]
        loss_p = -tf.reduce_mean(self.qvalue) + \
            tf.add_n([l2_p * tf.nn.l2_loss(wt) for wt in wts_p])
        grads_p = tf.gradients(loss_p, vars_p)
        if max_grad_norm > 0:
            grads_p, grad_norm_p = tf.clip_by_global_norm(grads_p, max_grad_norm)
        self.lr_p = tf.placeholder(tf.float32, [])
        optimizer_p = tf.train.AdamOptimizer(learning_rate=self.lr_p)
        self._train_p = optimizer_p.apply_gradients(list(zip(grads_p, vars_p)))

        # target nets update
        vars_vtar = tf.trainable_variables(scope=self.name + '_tarcritic')
        vars_ptar = tf.trainable_variables(scope=self.name + '_taractor')
        _init_ptar, _update_ptar = self._init_target_update(vars_p, vars_ptar, tau)
        _init_vtar, _update_vtar = self._init_target_update(vars_v, vars_vtar, tau)
        self.init_target = [_init_ptar, _init_vtar]
        self.update_target = [_update_ptar, _update_vtar]
        loss = loss_v * v_coef + loss_p

        summaries = []
        summaries.append(tf.summary.scalar('loss/policy_loss', loss_p))
        summaries.append(tf.summary.scalar('loss/value_loss', loss_v))
        summaries.append(tf.summary.scalar('loss/total_loss', loss))
        summaries.append(tf.summary.scalar('train/policy_gradnorm', grad_norm_p))
        summaries.append(tf.summary.scalar('train/value_gradnorm', grad_norm_v))
        self.summary = tf.summary.merge(summaries)


class DDPGFCPolicy(DDPGPolicy):
    """
    A DDPG policy using a DNN with all layers fully-connected.

    Attributes:
        name (str): policy name
        S (tf placeholder): MDP state
        A (tf placeholder): MDP action
        S1 (tf placeholder): MDP next state
        R (tf placeholder): MDP reward
        DONE (tf placeholder): MDP done flag
        action (tf node): output forward action
        qvalue (tf node): output Q-value (actor trainable) for backward actor update
        q0 (tf node): output Q-value (actor frozen) for backward critic update
        q1 (tf node): output next Q-value (actor greedy) for backward critic update
    """
    def __init__(self, n_s, n_a, n_step, n_fc, noise_generator):
        """Policy initialization."""
        super().__init__(n_a, n_s, n_step, noise_generator)
        self.name = 'ddpgfc'
        self._init_graph(n_fc)

    def backward(self, sess, obs, acts, next_obs, dones, rs,
                 cur_lr_v, cur_lr_p, warmup=False, summary_writer=None,
                 global_step=None):
        """Backward update on actor and critic."""
        train_op = [self._train_v]
        if not warmup:
            train_op.append(self._train_p)
        if summary_writer is not None:
            train_op.append(self.summary)
        outs = sess.run(train_op,
                       {self.S: obs,
                        self.A: acts,
                        self.S1: next_obs,
                        self.R: rs,
                        self.DONE: dones,
                        self.lr_v: cur_lr_v,
                        self.lr_p: cur_lr_p})
        sess.run(self.update_target)
        if summary_writer is not None:
            summary_writer.add_summary(outs[-1], global_step=global_step)

    def forward(self, sess, ob, mode='explore'):
        """Forward inference on actor."""
        action = sess.run(self.action, {self.S: np.array(ob)})
        if mode != 'explore':
            return np.squeeze(action)
        return np.clip(np.squeeze(action) + self.noise_generator.noise(), -1, 1)

    def _build_q_net(self, s, a, n_fc):
        h = rl_utils.fc(s, 'fcs', n_fc[0])
        h = tf.concat([h, a], 1)
        return self._build_fc_net(h, n_fc[1:], 'v')

    def _init_graph(self, n_fc):
        self.S = tf.placeholder(tf.float32, [None, self.n_s])
        self.A = tf.placeholder(tf.float32, [self.n_step, self.n_a])
        self.S1 = tf.placeholder(tf.float32, [self.n_step, self.n_s])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.DONE = tf.placeholder(tf.bool, [self.n_step])
        # actor action and qvalue
        with tf.variable_scope(self.name + '_actor'):
            self.action = self._build_fc_net(self.S, n_fc, 'pi')
            # self.exploration = tf.squeeze(self.action, axis=0) + self.gen_noise
        with tf.variable_scope(self.name + '_critic'):
            self.qvalue = self._build_q_net(self.S, self.action, n_fc)

        # batch action and qvalue
        with tf.variable_scope(self.name + '_critic', reuse=True):
            self.q0 = self._build_q_net(self.S, self.A, n_fc)
        # Tq is estimated by target nets, as ground truth of critic
        with tf.variable_scope(self.name + '_taractor'):
            a1 = self._build_fc_net(self.S1, n_fc, 'pi')
        with tf.variable_scope(self.name + '_tarcritic'):
            self.q1 = self._build_q_net(self.S1, a1, n_fc)

