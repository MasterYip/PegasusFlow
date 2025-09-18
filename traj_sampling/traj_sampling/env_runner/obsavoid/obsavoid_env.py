
from typing import overload
import numpy as np
import time
from matplotlib import pyplot as plt


class Obstacle1dEnv(object):
    """Obstacle environment
    Simple 1d environment which has only 1 dimension.
    Feasible region are defined as $F:{x|x \in [lb1, ub1] U [lb2, ub2] ...}$

    Args:
        object (_type_): _description_
    """

    def __init__(self, y=0.0, v=0.0, env_step=0.01, vis=True):
        # state
        self.y = y
        self.v = v
        self.t = 0
        self.y_hist = []

        # Configs
        self.hist_len = 1000

        # Default Obs Point Param
        self.obs_pt_param = [6, 5, 0.05, 0.3]

        # Acc Scaling: scaling for acceleration action input
        self.acc_scale = 1
        self.acc_bd = [-500, 500]
        # Vel Scaling: scaling for velocity observation output
        self.vel_scale = 10
        self.dy_scale = 0.001

        # PID
        self.p = 10000
        self.d = 200

        # config
        self.env_step = env_step
        self.bounds = []

        # vis
        self.enable_vis = vis
        self.vis_time_window = 1
        self.bd_vis_sample = 100
        if self.enable_vis:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.show(block=False)

    def set_pd(self, p, d):
        self.p = p
        self.d = d

    # Config
    def add_boundfunc(self, lb_fun, ub_fun):
        self.bounds.append((lb_fun, ub_fun))

    def step_env(self, act=0.0, vis=None):
        """Step env
        Args:
            act (float, optional): Acc scaled. Defaults to 0.0.
        """
        self.t += self.env_step
        self.y += self.v * self.env_step
        self.v += act * self.env_step * self.acc_scale
        self.y_hist.append(self.y)
        if len(self.y_hist) > self.hist_len:
            self.y_hist.pop(0)
        if vis is not False and self.enable_vis:
            self.vis_step()

    def step_env_y(self, y, vis=None):
        """Step env with y
        Args:
            y (float): y
        """
        self.t += self.env_step
        self.y = float(y)
        self.y_hist.append(self.y)
        if len(self.y_hist) > self.hist_len:
            self.y_hist.pop(0)
        if vis is not False and self.enable_vis:
            self.vis_step()

    def step_env_dy(self, dy, vis=None):
        """Step env with dy
        Args:
            dy (float): dy
        """
        self.t += self.env_step
        self.y += float(dy) * self.dy_scale
        self.y_hist.append(self.y)
        if len(self.y_hist) > self.hist_len:
            self.y_hist.pop(0)
        if vis is not False and self.enable_vis:
            self.vis_step()

    # Reference
    def sdf_value(self, y, t=None):
        values = []
        if not t:
            t = self.t
        for bd in self.bounds:
            if y > (bd[0](t)+bd[1](t))/2:
                values.append(bd[1](t) - y)
            else:
                values.append(y - bd[0](t))
        return max(values) if values else 0

    def get_ref_target(self):
        """Heuristic target for PID
        TODO: Too simple
        """
        bd = self.bounds[0]
        return (bd[0](self.t)+bd[1](self.t))/2

    def pid_ctrl(self):
        target = self.get_ref_target()
        return self.p*(target-self.y) + self.d*(-self.v)

    # Dataset
    def get_obspts(self):
        nt, ny, step, stepy = self.obs_pt_param
        if step is None:
            step = self.env_step
        obspts = []
        for i in range(ny):
            obspts += [[self.y + ((ny-1)/2-i)*stepy, self.t + j*step] for j in range(nt)]
        return obspts

    def get_observation(self):
        """Observation is state(2) + sdf_obs
        """
        state = [self.y, self.v/self.vel_scale]

        # next n sdf_value
        # sdf_obs = [self.sdf_value(self.y, self.t + i*step) for i in range(n)]
        sdf_obs = [self.sdf_value(y, t) for y, t in self.get_obspts()]
        return state + sdf_obs

    def get_action(self):
        acc = self.pid_ctrl()/self.acc_scale
        if acc > self.acc_bd[1]:
            acc = self.acc_bd[1]
        elif acc < self.acc_bd[0]:
            acc = self.acc_bd[0]
        return [acc]

    def get_action_y(self):
        return [self.y]

    def get_action_dy(self):
        if len(self.y_hist) < 2:
            return [0]
        return [(self.y - self.y_hist[-2])/self.dy_scale]

    def get_noised_action(self, noise=1.0):
        return [self.pid_ctrl()/self.acc_scale + np.random.randn()*noise]

    def get_reward(self):
        # return [self.sdf_value(self.y)]
        return self.sdf_value(self.y)

    # Vis
    def vis_step(self):
        self.ax.clear()
        self.ax.plot(np.linspace(self.t-self.env_step*len(self.y_hist), self.t, len(self.y_hist)), self.y_hist)
        X = np.linspace(self.t-self.vis_time_window, self.t+self.vis_time_window, self.bd_vis_sample)
        for bd in self.bounds:
            self.ax.plot(X, bd[0](X), 'r')
            self.ax.plot(X, bd[1](X), 'g')
        self.ax.set_ylim(-4, 4)
        self.ax.set_xlim(self.t-self.vis_time_window, self.t+self.vis_time_window)
        plt.pause(0.00001)

    def vis_rollout(self, actions, ctrl_mode='acc'):
        # Cache current state
        cached_state = [self.y, self.v, self.t, self.y_hist.copy()]
        Y = []
        T = []
        for act in actions:
            if ctrl_mode == 'acc':
                self.step_env(act, vis=False)
            elif ctrl_mode == 'y':
                self.step_env_y(act, vis=False)
            elif ctrl_mode == 'dy':
                self.step_env_dy(act, vis=False)
            Y.append(self.y)
            T.append(self.t)
        self.vis_scatter(T, Y)
        # Restore state
        self.y, self.v, self.t, self.y_hist = cached_state

    def vis_scatter(self, T, Y):
        # scatter marker size is 1
        self.ax.scatter(T, Y, s=1)
        plt.pause(0.00001)

    def end(self):
        if self.enable_vis:
            plt.close()


def sine_bound_env(vis=True, y=0.0, v=0.0, env_step=0.01):
    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)
    # env.add_boundfunc(lambda t: 0.5*np.sin(5*t)-0.2, lambda t: 0.5*np.sin(5*t)+0.2+0.1*np.sin(30*t))
    env.add_boundfunc(lambda t: 0.5*np.sin(15*t)-0.3, lambda t: 0.5*np.sin(15*t)+0.3)
    # env.add_boundfunc(lambda t: 0.5*np.sin(15*t)-0.2-1.5, lambda t: 0.5*np.sin(15*t)+0.2-1.5)
    return env


def increase_bound_env(vis=True, y=0, v=0, env_step=0.01):
    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)
    slope = 1.0
    env.add_boundfunc(lambda t: slope*t-0.2, lambda t: slope*t+0.2+0.1*np.sin(10*t))
    return env


def randpath_bound_env(vis=True, y=None, v=None, env_step=0.01):

    if y is None:
        y_bd = [-3, 3]
        y = np.random.rand()*(y_bd[1]-y_bd[0])+y_bd[0]
    if v is None:
        v_bd = [-5, 5]
        v = np.random.rand()*(v_bd[1]-v_bd[0])+v_bd[0]

    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)

    # p_bd = [1.5, 4]
    # p = 10**(np.random.rand()*(p_bd[1]-p_bd[0])+p_bd[0])
    # d_bd = [0.5, 2]
    # d = 10**(np.random.rand()*(d_bd[1]-d_bd[0])+d_bd[0])
    # env.set_pd(p, d)

    # wn_exp_bd = [0.8, 2]
    # zeta_bd = [0.3, 1.5]
    wn_exp_bd = [1.3, 1.8]
    zeta_bd = [0.6, 1.0]
    wn = 10**(np.random.rand()*(wn_exp_bd[1]-wn_exp_bd[0])+wn_exp_bd[0])
    zeta = np.random.rand()*(zeta_bd[1]-zeta_bd[0])+zeta_bd[0]
    p = wn**2
    d = 2*zeta*wn
    env.set_pd(p, d)
    print("wn: ", wn, "zeta: ", zeta)

    slope_abs_bd = 1.0
    coef_abs_bd = [1.0, 1.0, 0.5, 0.5, 0.3, 0.5, 0, 0.4, 0, 0.3, 0, 0, 0, 0, 0.1, 0.1]
    # coef_abs_bd = [1.0, 1.0, 0.5, 0.5]
    width_bd = [0.6, 1.5]
    slope = (np.random.rand()-0.5)*2*slope_abs_bd
    coef = [(np.random.rand()-0.5)*2*coef_abs_bd[i//2] for i in range(len(coef_abs_bd)*2)]
    width = np.random.rand()*(width_bd[1]-width_bd[0])+width_bd[0]

    def func(t):
        res = slope*t
        for i in range(len(coef_abs_bd)):
            res += coef[i*2]*np.sin((i+1)*t) + coef[i*2+1]*np.cos((i+1)*t)
        return res
    env.add_boundfunc(lambda t: func(t)-width/2, lambda t: func(t)+width/2)
    return env


def test_rand_bound_env():
    import numpy as np
    while True:
        env = randpath_bound_env()
        last_y = 0
        # stop when ctrl+c
        for i in range(100):
            # num = 30
            # fake_noise_scatter_X = np.linspace(env.t, env.t+0.3, num)
            # fake_noise_scatter_Y = env.y + np.random.randn(num)*0.1
            env.step_env(act=env.get_action()[0])
            # env.step_env(acc=env.get_noised_action()[0])
            # env.vis_scatter(fake_noise_scatter_X, fake_noise_scatter_Y)
            pts = env.get_obspts()
            T = [pt[1] for pt in pts]
            Y = [pt[0] for pt in pts]
            env.vis_scatter(T, Y)
            # print(env.sdf_value(env.y))
            # time.sleep(env.env_step)

            # # Print obs, action
            # print("obs: ", env.get_observation())
            # print("action: ", env.get_action())
            # print("action_y: ", env.y)
            # reserve up to 2 decimal places
            # print("obs", ["{:.2f}".format(num) for num in env.get_observation()])
            obs = env.get_observation()
            print("obs_y: ", ["{:.2f}".format(obs[0])])
            print("obs_v: ", ["{:.2f}".format(obs[1])])
            print("dy: ", ["{:.2f}".format(env.y - last_y)])
            print("action", ["{:.2f}".format(num) for num in env.get_action()])
            print("action_dy", ["{:.2f}".format(num) for num in env.get_action_dy()])

            sdf_obs = np.array(obs[2:]).reshape(5, 6)
            print("sdf_obs: ")
            for row in sdf_obs:
                print(["{:.2f}".format(num) for num in row])
                # print(["○" if num > 0 else "●" for num in row])

            # print("action_y: ", ["{:.2f}".format(env.y)])
            last_y = env.y
            # plt.pause(0.1)
        env.end()


if __name__ == "__main__":
    test_rand_bound_env()
