import os
import sys
import argparse
import importlib
import warnings

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import utils.import_envs  # pytype: disable=import-error
import numpy as np
import pickle
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset
from stable_baselines import PPO2, SAC

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from utils.utils import StoreDict

# Keras
import tensorflow as tf
import keras
from keras import Sequential, Model, Input
from keras.utils import to_categorical
from keras.layers import Dense, Flatten,multiply, Dropout, Reshape, Activation, Lambda, Dot
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import metrics
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import numpy as np

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def linear_vector_data(episodes, frame_num = 4, action_num = 2, use_images = False):
    stacks = []
    actions = []
    for j in range(len(episodes)):
        t_f = []
        a_f = []
        frames, inputs, rewards, _ = zip(*episodes[j])

        for i in range(len(episodes[j])-frame_num):
            if use_images:
                t_f.append(np.concatenate(np.array(frames[i:i + frame_num]), axis=2))
            else:
                t_f.append(np.concatenate(np.array(frames[i:i+frame_num])))
                a_f.append(inputs[i])

        stacks += t_f
        actions += a_f

    return np.array(stacks), np.array(actions)


def inverse_vector_data(episodes, frame_num = 4, action_num = 2, use_images = False):
    stacks = []
    actions = []
    for j in range(len(episodes)):
        t_f = []
        a_f = []
        frames, inputs, _, _ = zip(*episodes[j])

        for i in range(len(episodes[j]) - frame_num):
            if use_images:
                t_f.append(np.concatenate(np.array(frames[i:i + frame_num]), axis=2))
            else:
                t_f.append(np.concatenate(np.array(frames[i:i+frame_num])))
                a_f.append(inputs[i])

        stacks += t_f
        #actions += inputs[frame_num-2:-1]
        actions += a_f

    return np.array(stacks), np.array(actions)


# Non-Linear Vector BCO
def linear_clone_model_continuous(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, action_num = 2):

        image = Input(shape = (frame_num * dim,), name='image')
        x = Dense(300, activation = 'relu')(image)
        x = Dense(200, activation='relu')(x)
        action = Dense(action_num, activation='linear')(x)
        model = Model(inputs=[image], outputs=[action])

        model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model

def linear_inverse_model_continuous(learning_rate=0.001, decay=0.0, dim=4, frame_num=4, action_num=2):
    image = Input(shape=(frame_num * dim,), name='image')
    x = Dense(300, activation='relu')(image)
    x = Dense(200, activation='relu')(x)
    action = Dense(action_num, activation='linear')(x)
    model = Model(inputs=[image], outputs=[action])

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay),
                  metrics=['accuracy'])
    return model


def train_clone_continuous(c_model, i_model, model_path, data_path, bco_episodes = [], epoch = 30, data_size=30, pre_train_data = 1, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):

    # Split data into labeled/unlabeled
    if indices == []:
        indices = np.random.choice(data_size, pre_train_data, replace=False)
    dual_indices = [i for i in range(pre_train_data) if i not in indices]

    episodes = load_episodes(data_path, indices)
    data, actions = linear_vector_data(episodes,frame_num = frame_num, action_num = action_num, use_images = use_images)

    data = data.reshape((data.shape[0], -1))
    actions = actions.reshape((actions.shape[0], -1))

    # Load unlabeled data and label with inverse model
    if not pretrain:
        episodes = load_episodes(data_path, dual_indices)
        n_data, n_actions = inverse_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
        n_data = n_data.reshape((n_data.shape[0], -1))
        n_actions = n_actions.reshape((n_actions.shape[0], -1))
        n_actions = i_model.predict(n_data)
        n_data, _ = linear_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
        n_data = n_data.reshape((n_data.shape[0], -1))
        print(n_data.shape)
        print(n_actions.shape)
        data = np.concatenate([data,n_data],axis = 0)
        actions = np.concatenate([actions,n_actions])

    print("Training with " + str(len(actions)) + " samples")

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=False, patience=500)
    mc = ModelCheckpoint('c_model.h5', monitor='val_loss', mode='min', verbose=False, save_best_only=True)
    c_model.fit([data], actions, validation_split=0.3, batch_size=128, epochs=epoch, shuffle=True, verbose = verbose, callbacks=[es, mc])

    # serialize model to JSON
    model_json = c_model.to_json()
    with open(model_path + "c_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #c_model.save_weights(model_path + "c_model.h5")

    c_model.load_weights('c_model.h5')

    return c_model, indices


def train_inverse_continuous(i_model, model_path, data_path, bco_episodes = [], bco_size = 30, epoch=30, data_size=30, pre_train_data = 1, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):
    # Load all data
    if indices == []:
        indices = np.random.choice(data_size, pre_train_data, replace=False)
    episodes = load_episodes(data_path, indices)
    data, actions = inverse_vector_data(episodes, frame_num=frame_num, action_num=action_num,
                                             use_images=use_images)

    # Random sample up to data_size
    data = data.reshape((data.shape[0], -1))
    actions = actions.reshape((actions.shape[0], -1))
    # Concat
    if len(bco_episodes) != 0:
        # Load rollout data
        r_data, r_actions = inverse_vector_data(bco_episodes[:bco_size], frame_num=frame_num,
                                                     action_num=action_num,
                                                     use_images=use_images)
        r_data = r_data.reshape((r_data.shape[0], -1))
        r_actions = r_actions.reshape((r_actions.shape[0], -1))
        if len(data) != 0:
            data = np.concatenate([data, r_data])
            actions = np.concatenate([actions, r_actions])
        else:
            data = r_data
            actions = r_actions

    print("Training with " + str(len(actions)) + " samples")

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=False, patience=500)
    mc = ModelCheckpoint('i_model.h5', monitor='val_loss', mode='min', verbose=False, save_best_only=True)
    i_model.fit([data], [actions], validation_split=0.3, batch_size=128, epochs=epoch, shuffle=True, verbose = verbose, callbacks=[es, mc])

    # serialize model to JSON
    model_json = i_model.to_json()
    with open(model_path + "i_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #i_model.save_weights(model_path + "i_model.h5")

    i_model = load_model('i_model.h5')

    return i_model, indices


def save_episodes(dir, eps, num=0):
    for i in range(num, num + len(eps)):
        pickle.dump(eps[i], open(dir + str(i) + '.dump', 'wb'))
    return


# Load episodes from directory. You'll have to which ones you want.
def load_episodes(dir, nums):
    eps = []
    for i in nums:
        eps.append(pickle.load(open(dir + str(i) + '.dump', 'rb')))
    return eps


def evaluate(model, num_episodes = 100):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='AntBulletEnv-v0')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='sac',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                        type=int)
    parser.add_argument('--log-rollouts', help='Save Expert Trajectory Data', default=None,
                        type=str)
    parser.add_argument('--n-episodes', help='How Many Episdoes to Rollout', default=num_episodes,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=True,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--load-best', action='store_true', default=False,
                        help='Load best model instead of last model if available')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=np.random.randint(0,1000))
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)


    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id, load_best=args.load_best)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    log_dir = args.reward_log if args.reward_log != '' else None

    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    #hyperparams = OrderedDict([('batch_size', 256), ('buffer_size', 1000000), ('ent_coef', 'auto'), ('gamma', 0.98), ('gradient_steps', 1), ('learning_rate', 0.0003), ('learning_starts', 10000), ('n_timesteps', 2000000.0), ('policy', 'CustomSACPolicy'), ('tau', 0.01), ('train_freq', 1), ('normalize', False)])
    #stats_path = "trained_agents/sac/AntBulletEnv-v0"
    env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=not args.no_render,
                          hyperparams=hyperparams, env_kwargs=env_kwargs)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env

    '''model = ALGOS[algo].load(model_path, env=load_env)

    #generate_expert_traj(model, './Ant_Test', n_episodes=10)
    #dataset = ExpertDataset(expert_path='Ant_Test.npz', traj_limitation=-1, batch_size=128)

    #model = SAC('MlpPolicy', env = env, verbose=1)
    # Pretrain the PPO2 model
    print(env_id)
    #model.pretrain(dataset, n_epochs=1000)
    #model.save("ant_test")
    model = SAC.load("ant_test", env = env)

    #model.learn(total_timesteps=int(1.1e6), log_interval=10)
    #model.save("ant_test")
    #obs = env.reset()
    #print(obs.shape)'''

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic
    obs = env.reset()

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    state = None
    episodes = []
    for _ in range(args.n_episodes):
        # Setup Episode Recording
        #env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
        #                      stats_path=stats_path, seed=args.seed, log_dir=log_dir,
        #                      should_render=not args.no_render,
        #                      hyperparams=hyperparams, env_kwargs=env_kwargs)
        #obs = env.reset()
        #print(obs.shape)
        done = False
        episode = []
        episode.append([obs, 0, done, None])
        while not done:
            #action, state = model.predict(obs, state=state, deterministic=deterministic)
            #dist = model.predict([obs])[0]
            action = model.predict(obs)
            # Random Agent
            # action = [env.action_space.sample()]
            # Clip Action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)
            # If Logging Update
            episode[-1].insert(1, action)
            episode.append([obs, reward, done, infos])
            if not args.no_render:
                env.render('human')

            episode_reward += reward[0]
            ep_len += 1

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get('episode')
                    if episode_infos is not None:
                        print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                        print("Atari Episode Length", episode_infos['l'])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    #print("Episode Reward: {:.2f}".format(episode_reward))
                    #print("Episode Length", ep_len)
                    state = None
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0

                # Reset also when the goal is achieved when using HER
                if done or infos[0].get('is_success', False):
                    if args.algo == 'her' and args.verbose > 1:
                        print("Success?", infos[0].get('is_success', False))
                    # Alternatively, you can add a check to wait for the end of the episode
                    # if done:
                    obs = env.reset()
                    if args.algo == 'her':
                        successes.append(infos[0].get('is_success', False))
                        episode_reward, ep_len = 0.0, 0
        episodes.append(episode)

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))

    if args.verbose > 0 and len(episode_lengths) > 0:
        print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()

    if args.log_rollouts is not None:
        save_episodes(args.log_rollouts, episodes)

    return np.mean(episode_rewards), episodes


if __name__ == '__main__':
    learning_rates = [0.001, 0.001]
    dim = 29
    frame_num = 1
    action_num = 8

    pre_train_data = 3
    unlabeled_data = 30
    post_train_data = 15
    episodes = []

    i_model = linear_inverse_model_continuous(learning_rate=learning_rates[0], dim=dim, frame_num=frame_num,
                                              action_num=action_num)
    c_model = linear_clone_model_continuous(learning_rate=learning_rates[1], dim=dim, frame_num=frame_num,
                                            action_num=action_num)

    i_model, indices = train_inverse_continuous(i_model, "./Basic_Models/", "./Reacher_Vector_Rollouts_trash/",
                                                data_size=unlabeled_data, pre_train_data = pre_train_data,
                                                epoch=1000, frame_num=frame_num, action_num=action_num,
                                                use_images=False,
                                                verbose=False)

    model, _ = train_clone_continuous(c_model, i_model, "./Basic_Models/", "./Ant_Vector_Rollouts_trash/", epoch=1000,
                                      data_size=unlabeled_data, pre_train_data=pre_train_data, frame_num=frame_num,
                                      action_num=action_num, use_images=False, verbose=False, pretrain=False,
                                      indices=indices)

    mean_episode_reward_i, n_episodes = evaluate(model, num_episodes = 100)

    count = 0
    temp = [mean_episode_reward_i]
    while count < 3:
        print("Using " + str(count) + " rollouts")
        episodes += n_episodes[:20]
        count += 1
        post_train_data = count * 20
        i_model, _ = train_inverse_continuous(i_model, "./Basic_Models/", "./Ant_Vector_Rollouts_trash/",
                                                    data_size=unlabeled_data, pre_train_data=pre_train_data,
                                                    epoch=1000, frame_num=frame_num, action_num=action_num, bco_episodes = episodes, bco_size = post_train_data,
                                                    use_images=False,
                                                    verbose=False, indices = indices)

        model, _ = train_clone_continuous(c_model, i_model, "./Basic_Models/", "./Ant_Vector_Rollouts_trash/", epoch=1000,
                                          data_size=unlabeled_data, pre_train_data=pre_train_data, frame_num=frame_num,
                                          action_num=action_num, use_images=False, verbose=False, pretrain=False,
                                          indices=indices)

        mean_episode_reward_f, episodes = evaluate(model, num_episodes = 100)
        temp.append(mean_episode_reward_f)
    print(temp)

    print("Initial Reward: " + str(mean_episode_reward_i))
    print("Final Reward: " + str(mean_episode_reward_f))

