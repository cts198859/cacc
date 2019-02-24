"""
This module contains all helper functions for training and testing.
@author: Tianshu Chu
"""

import itertools
import logging
import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes()


def check_dir(cur_dir):
    """
    Check if the directory exists.

    Args:
        cur_dir (str): directory
    Returns:
        exists (bool): if it exists or not
    """
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    """
    Copy files.

    Args:
        src_dir (str): source file path
        tar_dir (str): target file path
    """
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    """
    Find first file with given extension under a given directory.

    Args:
        cur_dir (str): directory
        suffix (str): extension
    Returns:
        file_path (str): file path, return None if not found
    """
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    """
    Initialize sub-directories under base directory.

    Args:
        base_dir (str): base directory
        pathes ([str]): sub-directory names
    Returns:
        sub_dirs ({str}): dict of sub-directory name: path
    """
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    """Initialize log file under log_dir."""
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def print_log(s, level='info'):
    """Print logs at given level."""
    if level == 'info':
        logging.info(s)
    elif level == 'error':
        logging.error(s)


class Counter:
    """
    Counter of training steps.

    Attributes:
        counter: the actual counter
        cur_step: current step
        log_step: log interval
        test_step: test interval
        total_step: total training step
    """
    def __init__(self, total_step, test_step, log_step):
        """Initialization."""
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step

    def next(self):
        """Get and return the next step."""
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        """If the current model should be tested, based on test interval."""
        return ((self.cur_step + 1) % self.test_step == 0)

    def should_log(self):
        """If the current MDP step should be logged, based on log interval."""
        return ((self.cur_step + 1) % self.log_step == 0)

    def should_stop(self):
        """If the training should be stopped."""
        if self.cur_step >= self.total_step:
            return True
        return False


class Trainer():
    """
    Trainer manages the interaction between DRL agent and environment during the training.
    
    Attributes:
        cur_step (int): current step
        data ([dict]): list of training logs
        global_counter (Counter): step counter
        env (env.TrafficSimulator): environment simulator
        model (models.DDPG): DRL agent
        output_path (str): data output path
        sess: tf session
        summary_writer: tf summary writer
    """
    def __init__(self, env, model, global_counter, summary_writer, output_path=None):
        """Initialization."""
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.model = model
        self.sess = self.model.sess
        self.summary_writer = summary_writer
        self.data = []
        self.output_path = output_path
        self._init_summary()

    def _init_summary(self):
        """Event summary initialization."""
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        """Add values to event summary."""
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, ob):
        """
        Args:
            ob (array): MDP state
        Returns:
            next_ob (array): next MDP state
            reward (float): MDP reward
            done (bool): MDP done flag
        """
        action = self.model.forward([ob], mode='explore')
        next_ob, reward, done = self.env.step(action)
        # advance counter
        global_step = self.global_counter.next()
        self.model.add_transition(ob, action, reward, next_ob, done)
        # logging
        if self.global_counter.should_log():
            logging.info('''Training: global step %d, episode step %d,
                               ob: %s, a: %s, r: %s, done: %r''' %
                         (global_step, self.env.t,
                          str(ob), str(action), str(reward), done))
        return next_ob, reward, done

    def perform(self):
        """
        One-episode DRL inference in environment to collect evaluation performance.
        Note this function is similar to explore(self, ob, warmup), w/o the exploration noise.
        Returns:
            mean_reward (float): avg episode return
            std_reward (float): std episode return
        """
        ob = self.env.reset()
        if self.env.mode > 0:
            self.model.reset_noise()
        rewards = []
        while True:
            if self.env.mode > 0:
                action = self.model.forward([ob], mode='act')
                next_ob, reward, done = self.env.step(action)
            else:
                next_ob, reward, done = self.env.step()
            rewards.append(reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        return mean_reward, std_reward

    def plot_data(self, df):
        fig = plt.figure(figsize=(10, 8))
        df_train = df[df.mode == 'train']
        ts = df_train.step.values
        vmeans = df_train.avg_reward.values
        vstds = df_train.std_reward.values
        plt.plot(ts, vmeans, 'b', linewidth=3)
        plt.fill_between(ts, vmeans-vstds, vmeans+vstds, facecolor='b',
                         edgecolor='none', alpha=0.2)
        plt.grid(True, which='both')
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.xlabel('Training step', fontsize=15)
        plt.ylabel('Average episode reward', fontsize=15)
        plt.tight_layout()
        fig.savefig(self.output_path + 'train_reward_plot.csv')
        plt.close()

    def run(self):
        """
        Run DRL model training, and output the training logs.
        """
        while not self.global_counter.should_stop():
            # evaluate the model with certain intervals
            if self.global_counter.should_test():
                global_step = self.global_counter.cur_step
                self.env.train = False
                # in evaluation we do not explore
                mean_reward, std_reward = self.perform()
                log = {
                       'step': global_step,
                       'mode': 'test',
                       'avg_reward': mean_reward,
                       'std_reward': std_reward}
                self.data.append(log)
                self._add_summary(mean_reward, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f, std R: %.2f' %
                             (global_step, mean_reward, std_reward))

            # one-episode training
            self.env.train = True
            # reset environment
            ob = self.env.reset()
            # reset OU noise
            self.model.reset_noise()
            rewards = []
            while True:
                # one-step forward exploration
                ob, reward, done = self.explore(ob)
                rewards.append(reward)
                global_step = self.global_counter.cur_step
                # one-step backward update
                self.model.backward(self.summary_writer, global_step)
                # episode termination
                if done:
                    break
            # append training logs
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            log = {
                   'step': global_step,
                   'mode': 'train',
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            # flush tensorboard every training episode
            self.summary_writer.flush()

        # output logs to a csv table
        df = pd.DataFrame(self.data)
        self.plot_data(df)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    """
    Tester manages the interaction between DRL agent and environment during the evaluation.
    It is similar to Trainer, but enabling environment to record more data.

    Attributes:
        data ([dict]): list of testing logs
        env (env.TrafficSimulator): environment simulator
        model (models.DDPG): pre-trained DRL agent
        output_path (str): data output path
    """
    def __init__(self, env, model, output_path):
        """Initialzation."""
        self.env = env
        self.model = model
        self.env.train = False
        self.output_path = output_path

    def run(self):
        """
        Run DRL model evaluation, and output rich testing logs.
        """
        mean_reward, std_reward = self.perform()
        logging.info('Offline testing: avg R: %.2f, std R: %.2f' % (mean_reward, std_reward))
        self.env.output_data(self.output_path)
