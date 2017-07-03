#!/usr/bin/env python
"""Example code of DDPG on OpenAI Gym environments.

For DDPG, see: https://arxiv.org/abs/1509.02971
"""
from __future__ import print_function
from __future__ import division
import argparse
import collections
import copy
import random

import gym
import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers


class QFunction(chainer.Chain):
    """Q-function represented by a MLP."""

    def __init__(self, obs_size, action_size, n_units=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size + action_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, 1,
                               initialW=chainer.initializers.HeNormal(1e-3))

    def __call__(self, obs, action):
        """Compute Q-values for given state-action pairs."""
        x = F.concat((obs, action), axis=1)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)


def squash(x, low, high):
    """Squash values to fit [low, high] via tanh."""
    center = (high + low) / 2
    scale = (high - low) / 2
    return F.tanh(x) * scale + center


class Policy(chainer.Chain):
    """Policy represented by a MLP."""

    def __init__(self, obs_size, action_size, action_low, action_high,
                 n_units=100):
        super(Policy, self).__init__()
        self.action_high = action_high
        self.action_low = action_low
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, action_size,
                               initialW=chainer.initializers.HeNormal(1e-3))

    def __call__(self, x):
        """Compute actions for given observations."""
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return squash(self.l2(h),
                      self.xp.asarray(self.action_low),
                      self.xp.asarray(self.action_high))


def get_greedy_action(policy, obs):
    obs = policy.xp.asarray(obs[None], dtype=np.float32)
    with chainer.no_backprop_mode():
        action = policy(obs).data[0]
    return chainer.cuda.to_cpu(action)


def update(Q, target_Q, policy, target_policy, opt_Q, opt_policy,
           samples, gamma=0.99, target_type='double_dqn'):
    n = len(samples)
    xp = Q.xp
    s = xp.asarray([sample[0] for sample in samples], dtype=np.float32)
    a = xp.asarray([sample[1] for sample in samples], dtype=np.float32)
    r = xp.asarray([sample[2] for sample in samples], dtype=np.float32)
    done = xp.asarray([sample[3] for sample in samples], dtype=np.float32)
    s_next = xp.asarray([sample[4] for sample in samples], dtype=np.float32)

    def update_Q():
        # Predicted values: Q(s,a)
        y = F.reshape(Q(s, a), (n,))
        # Target values: r + gamma * Q(s,policy(s))
        with chainer.no_backprop_mode():
            next_q = F.reshape(target_Q(s_next, target_policy(s_next)), (n,))
            t = r + gamma * (1 - done) * next_q
        loss = F.mean_squared_error(y, t)
        Q.cleargrads()
        loss.backward()
        opt_Q.update()

    def update_policy():
        # Maximize Q(s,policy(s))
        loss = - F.sum(Q(s, policy(s))) / n
        policy.cleargrads()
        loss.backward()
        opt_policy.update()

    update_Q()
    update_policy()


def soft_copy_params(source, target, tau):
    for s, t in zip(source.params(), target.params()):
        t.data[:] += tau * (s.data - t.data)


def main():

    parser = argparse.ArgumentParser(description='Chainer example: DRL(DDPG)')
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='Name of the OpenAI Gym environment to play')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of transitions in each mini-batch')
    parser.add_argument('--episodes', '-e', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='ddpg_result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--reward-scale', type=float, default=1e-3,
                        help='Reward scale factor')
    parser.add_argument('--replay-start-size', type=int, default=500,
                        help='Number of steps after which replay is started')
    parser.add_argument('--tau', type=float, default=1e-2,
                        help='Softness of soft target update')
    parser.add_argument('--noise-scale', type=float, default=0.4,
                        help='Scale of additive Gaussian noises')
    parser.add_argument('--record', action='store_true', default=True,
                        help='Record performance')
    parser.add_argument('--no-record', action='store_false', dest='record')
    args = parser.parse_args()

    # Initialize an environment
    env = gym.make(args.env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    ndim_action = env.action_space.low.size
    if args.record:
        env.monitor.start(args.out, force=True)

    # Initialize variables
    D = collections.deque(maxlen=10 ** 6)
    Rs = collections.deque(maxlen=100)
    step = 0

    # Initialize models and optimizers
    Q = QFunction(ndim_obs, ndim_action, n_units=args.unit)
    policy = Policy(ndim_obs, ndim_action,
                    env.action_space.low, env.action_space.high,
                    n_units=args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        Q.to_gpu(args.gpu)
        policy.to_gpu(args.gpu)
    target_Q = copy.deepcopy(Q)
    target_policy = copy.deepcopy(policy)
    opt_Q = optimizers.Adam()
    opt_Q.setup(Q)
    opt_policy = optimizers.Adam(alpha=1e-4)
    opt_policy.setup(policy)

    for episode in range(args.episodes):

        obs = env.reset()
        done = False
        R = 0.0
        t = 0

        while not done and t < env.spec.timestep_limit:

            # Select an action with additive noises for exploration
            a = (get_greedy_action(policy, obs) +
                 np.random.normal(scale=args.noise_scale))

            # Execute an action
            new_obs, r, done, _ = env.step(
                np.clip(a, env.action_space.low, env.action_space.high))
            R += r

            # Store a transition
            D.append((obs, a, r * args.reward_scale, done, new_obs))
            obs = new_obs

            # Sample a random minibatch of transitions and replay
            if len(D) >= args.replay_start_size:
                samples = random.sample(D, args.batchsize)
                update(Q, target_Q, policy, target_policy,
                       opt_Q, opt_policy, samples)

            # Soft update of the target networks
            soft_copy_params(Q, target_Q, args.tau)
            soft_copy_params(policy, target_policy, args.tau)

            step += 1
            t += 1

        Rs.append(R)
        average_R = np.mean(Rs)
        print('episode: {} step: {} R:{} average_R:{}'.format(
              episode, step, R, average_R))

    if args.record:
        env.monitor.close()


if __name__ == '__main__':
    main()
