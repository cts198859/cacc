"""
Main script for running the program, with arguments:
base-dir: the root directory of the experiment. Model, plot, and simulation data will be exported there.
config-dir: the path of the experiment config file, which defines all the hyper-parameters. An example 
config file can be found under ./configs/.
mode: train or test.

During training, the tensorboard is accessible under base-dir/log/. After training, the model is saved
under base-dir/model/ and the evaluation data is exported to base-dir/data/.
@author: Tianshu Chu
"""

import argparse
import configparser
import tensorflow as tf

import env
import models
import train_utils


def parse_args():
    """
    Parse commandline arguments.
    """
    default_base_dir = '/Users/tchu/Documents/rl_test/cacc_catchup0'
    default_config_dir = './configs/cacc_catchup0.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--mode', type=str, required=False,
                        default='train', help="train/test")
    return parser.parse_args()


def train(args):
    """
    Train DRL model using specified DRL algorithm (models.DDPG) and
    environment simulator (env.TrafficSimulator). It runs a train_utils.Trainer to
    train a DRL model, and run a train_utils.Tester to evaluate it afterwards.

    Args:
        args: parsed commandline arguments
    """
    # init experiment and log directories
    base_dir = args.base_dir
    dirs = train_utils.init_dir(base_dir)
    train_utils.init_log(dirs['log'])

    # copy and load config file
    config_dir = args.config_dir
    train_utils.copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env simulator
    simulator = env.CACC(config['ENV_CONFIG'])

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = train_utils.Counter(total_step, test_step, log_step)

    # init DDPG agent
    if simulator.mode == 1:
        model = models.DDPG(8*3, 4,
                            total_step=total_step, model_config=config['MODEL_CONFIG'])
        model.init_train()
    elif simulator.mode == 2:
        model = models.DDPG(8*3, 8,
                            total_step=total_step, model_config=config['MODEL_CONFIG'])
        model.init_train()
    else:
        model = None

    # init tf summary writer and trainer
    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = train_utils.Trainer(simulator, model, global_counter, summary_writer,
                                  output_path=dirs['data'])
    trainer.run()

    # save trained model
    final_step = global_counter.cur_step
    train_utils.print_log('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)

    # evaluate trained model
    tester = train_utils.Tester(simulator, model, dirs['data'])
    tester.run()


def test(args):
    """
    Evaluate pre-trained DRL model (models.DDPG) using specified
    environment simulator (env.TrafficSimulator), by running a train_utils.Tester.
    Args:
        args: parsed commandline arguments
    """
    # init experiment and log directories
    base_dir = args.base_dir
    dirs = train_utils.init_dir(base_dir, pathes=['eva_data', 'eva_log'])
    train_utils.init_log(dirs['eva_log'])

    # load config file and init simulator
    config = configparser.ConfigParser()
    config.read(args.config_dir)
    simulator = env.CACC(config['ENV_CONFIG'])

    # load pre-trained model
    if simulator.mode == 1:
        model_dir = base_dir + '/model'
        if not train_utils.check_dir(model_dir):
            train_utils.print_log('Evaluation: trained model does not exist!', level='error')
            return
        model = models.DDPG(8*3, 4, model_config=config['MODEL_CONFIG'])
        if not model.load(model_dir + '/'):
            return
    else:
        model = None

    # evaluate
    tester = train_utils.Tester(simulator, model, dirs['eva_data'])
    tester.run()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    else:
        test(args)
