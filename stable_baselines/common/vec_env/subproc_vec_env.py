import multiprocessing
from collections import OrderedDict

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images


def _worker(remote, parent_remote, env_fn_wrapper, n_envs):
    parent_remote.close()
    envs = [env_fn_wrapper.var() for _ in range(n_envs)]
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                # keep a list of tuples [(obs, reward, done, info), ...] instead of
                # combining; SubprocVecEnv will handle that part w/ all processes
                obs_rew_done_info = [list(env.step(datum)) for env, datum in zip(envs, data)]
                for i, env in enumerate(envs):
                    if obs_rew_done_info[i][2]:  # if done: obs = env.reset()
                        obs_rew_done_info[i][0] = env.reset()
                remote.send(obs_rew_done_info)
            elif cmd == 'reset':
                observations = [env.reset() for env in envs]
                remote.send(observations)
            elif cmd == 'render':
                remote.send([env.render(*data[0], **data[1]) for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == 'env_method':
                methods = [getattr(env, data[0]) for env in envs]
                remote.send([method(*data[1], **data[2]) for method in methods])
            elif cmd == 'get_attr':
                remote.send([getattr(env, data) for env in envs])
            elif cmd == 'set_attr':
                remote.send([setattr(env, data[0], data[1]) for env in envs])
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow
        sessions or other non thread-safe libraries are used in the parent (see issue #217).
        However, compared to 'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods,
        users must wrap the code in an ``if __name__ == "__main__":``
        For more information, see the multiprocessing documentation.

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'fork' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, n_envs_per_process=1, start_method=None):
        self.waiting = False
        self.closed = False
        self.n_envs_per_process = n_envs_per_process
        n_processes = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            fork_available = 'fork' in multiprocessing.get_all_start_methods()
            start_method = 'fork' if fork_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_processes)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), n_envs_per_process)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, n_processes * n_envs_per_process, observation_space, action_space)

    def step_async(self, actions):
        for i, remote in enumerate(self.remotes):
            remote.send(('step', actions[i * self.n_envs_per_process : (i + 1) * self.n_envs_per_process]))
        self.waiting = True

    def step_wait(self):
        results = [result for remote in self.remotes for result in remote.recv()]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [ob for remote in self.remotes for ob in remote.recv()]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [img for pipe in self.remotes for img in pipe.recv()]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [img for pipe in self.remotes for img in pipe.recv()]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        # TODO - I haven't done anything with indices; not sure if you'd want to be able to
        # index each environment in each process? Currently this only lets you index
        # the processes, so you get results from *all* environments in each process
        # (if n_process_per_env == 1 then the behavior is unchaged)
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [attr for remote in target_remotes for attr in remote.recv()]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [ret for remote in target_remotes for ret in remote.recv()]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
