"""
Training script for RARL framework
Reading and unstanding line by line
"""

import argparse
import logging
import PPO_RARL
import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import logger
from reimplementation import MlpPolicy
import numpy as np


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)


def train(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    num_iters = args.num_iters
    env = gym.make(args.env)
    # test_env = gym.make('CartPole-v2')
    test_env = gym.make(args.test_env)

    # env.seed(args.seed)
    # test_env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)

    pi, adv_pi = PPO_RARL.learn(env, test_env, policy_fn,
                         max_iters=num_iters,
                         timesteps_per_batch=2048,
                         clip_param=0.2, entcoeff=0.0,
                         optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                         gamma=0.99, lam=0.95, schedule=args.mode,
                         )
    env.close()

    # run trained model right away
    if args.play:
        logger.log("Running trained model")
        # TODO
        obs = test_env.reset()
        # obs = env.reset()
        while True:
            test_env.render()
            action = pi.act(False, obs)[0]
            action_adv = adv_pi.act(False, obs)[0]
            action = np.append(action,action_adv)
            obs, rew, done, _ = test_env.step(action)
            if done:
                print('episode is ending\n')
                obs = test_env.reset()
        test_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', default='CartPole-v0')
    parser.add_argument('--env', help='environment ID', default='CentralDecision-v0')
    parser.add_argument('--test_env', help='test environment', default='CentralDecisionTest-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)        # 千万别设置0，否则生成的随机数一样
    parser.add_argument('--play', help='Run trained model', type=int, default=None)
    parser.add_argument('--adversary', help='Numbers of adversary forces', type=int, default=1)
    parser.add_argument('--num_iters', help='Numbers of iterations', type=int, default=600)
    parser.add_argument('--mode', help='learning rate between two agent', default='linear')  # constant linear and asym
    args = parser.parse_args()
    train(args)
