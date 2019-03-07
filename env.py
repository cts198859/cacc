import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes()

HG_SCALE = 20

class CACC:
    def __init__(self, config):
        self._load_config(config)
        self.ovm = OVMCarFollowing(self.h_s, self.h_g, self.v_max)
        self.train = True

    def constrain_speed(self, v, u):
        # apply constraints
        v_next = v + np.clip(u, self.u_min, self.u_max) * self.dt
        v_next = np.clip(v_next, 0, self.v_max)
        u_const = (v_next - v) / self.dt
        return v_next, u_const

    def get_human_accel(self, i, h_g):
        v = self.vs_cur[i]
        h = self.hs_cur[i]
        if i:
            v_lead = self.vs_cur[i-1]
        else:
            v_lead = self.v0s[self.t]
        alpha = self.alphas[i]
        beta = self.betas[i]
        return self.ovm.get_accel(v, v_lead, h, alpha, beta, h_g)

    def get_reward(self):
        v_state = np.array(self.vs_cur, copy=True)
        h_state = np.array(self.hs_cur, copy=True)
        u_state = np.array(self.us_cur, copy=True)
        # give large penalty for collision
        if np.min(h_state) < self.h_min:
            return -self.G
        h_rewards = -(h_state - self.h_star) ** 2
        v_rewards = -self.a * (v_state - self.v_star) ** 2
        u_rewards = -self.b * (u_state) ** 2
        if self.train:
            c_rewards = self.c * (np.minimum(h_state - self.h_s, 0)) ** 2
        else:
            c_rewards = 0
        return np.mean(h_rewards + v_rewards + u_rewards + c_rewards)

    def get_state(self):
        if not len(self.auto_vehs):
            return None
        # find vehicles out of range of V2V communication
        invalid_vehs = []
        for i in range(self.n_veh-1):
            if i in self.auto_vehs:
                continue
            if min(self.hs_cur[i], self.hs_cur[i+1]) > self.D:
                invalid_vehs.append(i)
        v_state = np.array(self.vs_cur, copy=True)
        h_state = np.array(self.hs_cur, copy=True)
        u_state = np.array(self.us_cur, copy=True)
        # disable out-of-range vehicle states
        for i in invalid_vehs:
            v_state[i] = 0
            h_state[i] = 0
            u_state[i] = 0
        # normalize state
        v_state = np.clip((v_state - self.v_star) / 5, -2, 2)
        h_state = np.clip((h_state - self.h_star) / 10, -2, 2)
        u_state = u_state / self.u_max
        return np.concatenate([h_state, v_state, u_state])

    def output_data(self, path):
        hs = np.array(self.hs)
        vs = np.array(self.vs)
        us = np.array(self.us)
        df = pd.DataFrame()
        df['time'] = np.arange(len(hs)) * self.dt
        df['reward'] = np.array(self.rewards)
        for i in range(self.n_veh):
            df['headway_%d' % (i+1)] = hs[:, i]
            df['velocity_%d' % (i+1)] = vs[:, i]
            df['control_%d' % (i+1)] = us[:, i]
        df.to_csv(path + 'env_data.csv')
        self.plot_data(df, path)

    def plot_data(self, df, path):
        fig = plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        for i in [0, 2, 5, 7]:
            plt.plot(df.time.values, df['headway_%d' % (i+1)].values, linewidth=3,
                     label='veh #%d' % (i+1))
        plt.legend(fontsize=20, loc='best')
        plt.grid(True, which='both')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Headway [m]', fontsize=20)
        plt.subplot(2, 1, 2)
        for i in [0, 2, 5, 7]:
            plt.plot(df.time.values, df['velocity_%d' % (i+1)].values, linewidth=3,
                     label='veh #%d' % (i+1))
        # plt.legend(fontsize=15, loc='best')
        plt.grid(True, which='both')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Velocity [m/s]', fontsize=20)
        plt.xlabel('Time [s]', fontsize=20)
        fig.tight_layout()
        plt.savefig(path + 'env_plot.pdf')
        plt.close()

    def reset(self, h0=None, v0=None):
        self._init_common()
        if self.scenario.startswith('catchup'):
            self._init_catchup()
        elif self.scenario.startswith('slowdown'):
            self._init_slowdown()
        self.hs_cur = self.hs[0]
        self.vs_cur = self.vs[0]
        if h0 is not None:
            self.hs_cur[0] = h0
        if v0 is not None:
            self.vs_cur[0] = v0
        self.us_cur = [0] * self.n_veh
        self.us = [self.us_cur]
        self.rewards = [0]
        return self.get_state()

    def step(self, action=0):
        auto_hgs = self.h_g + action * HG_SCALE
        hs_next = []
        vs_next = []
        self.us_cur = []
        # update speed
        for i in range(self.n_veh):
            if (self.mode == 1) and (i in self.auto_vehs):
                h_g = auto_hgs[self.auto_vehs.index(i)]
            else:
                h_g = -1
            u = self.get_human_accel(i, h_g)
            v_next, u_const = self.constrain_speed(self.vs_cur[i], u)
            self.us_cur.append(u_const)
            vs_next.append(v_next)
        # update headway
        for i in range(self.n_veh):
            if i == 0:
                v_lead = self.v0s[self.t]
                v_lead_next = self.v0s[self.t+1]
            else:
                v_lead = self.vs_cur[i-1]
                v_lead_next = vs_next[i-1]
            v = self.vs_cur[i]
            v_next = vs_next[i]
            hs_next.append(self.hs_cur[i] + 0.5*self.dt*(v_lead+v_lead_next-v-v_next))
        self.hs_cur = hs_next
        self.vs_cur = vs_next
        self.hs.append(self.hs_cur)
        self.vs.append(self.vs_cur)
        self.us.append(self.us_cur)
        self.t += 1
        reward = self.get_reward()
        self.rewards.append(reward)
        done = False
        if reward == -self.G:
            done = True
        if self.t == self.T:
            done = True
        return self.get_state(), reward, done

    def _init_catchup(self):
        # only the first vehicle has long headway
        self.hs = [[self.h_star*4] + [self.h_star] * 7]
        self.vs = [[self.v_star] * 8]
        self.v0s = [self.v_star] * (self.T+1)

    def _init_common(self):
        if self.mode == 0:
            self.auto_vehs = []
            self.human_vehs = list(range(8))
        elif self.mode == 1:
            self.auto_vehs = [0, 2, 4, 6]
            self.human_vehs = [1, 3, 5, 7]
        self.alphas = [0.4, 0.4, 0.4, 0.3, 0.4, 0.3, 0.4, 0.5]
        self.betas = [0.4, 0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.5]
        self.n_veh = 8
        self.t = 0

    def _init_slowdown(self):
        self.hs = [[self.h_star/3*4] * 8]
        self.vs = [[self.v_star*2] * 8]
        v0s_decel = list(np.arange(self.v_star*2, self.v_star-0.1, self.u_min/2))
        self.v0s = v0s_decel + [self.v_star] * (self.T+1-len(v0s_decel))

    def _load_config(self, config):
        self.T = config.getint('episode_length')
        self.dt = config.getfloat('delta_t')
        self.D = config.getfloat('communication_range')
        self.h_min = config.getfloat('headway_min')
        self.h_star = config.getfloat('headway_target')
        self.h_s = config.getfloat('headway_st')
        self.h_g = config.getfloat('headway_go')
        self.v_max = config.getfloat('speed_max')
        self.v_star = config.getfloat('speed_target')
        self.u_min = config.getfloat('accel_min')
        self.u_max = config.getfloat('accel_max')
        self.scenario = config.get('scenario')
        self.mode = int(self.scenario[-1])
        self.a = config.getfloat('reward_a')
        self.b = config.getfloat('reward_b')
        self.c = config.getfloat('reward_c')
        self.G = config.getfloat('penalty')


class OVMCarFollowing:
    '''
    A OVM controller for human-driven vehicles
    Attributes:
        h_st (float): stop headway
        h_go (float): full-speed headway
        v_max (float): max speed
    '''
    def __init__(self, h_st, h_go, v_max):
        """Initialization."""
        self.h_st = h_st
        self.h_go = h_go
        self.v_max = v_max

    def get_accel(self, v, v_lead, h, alpha, beta, h_go=-1):
        """
        Get target acceleration using OVM controller.

        Args:
            v (float): current vehicle speed
            v_lead (float): leading vehicle speed
            h (float): current headway
            alpha, beta (float): human parameters
        Returns:
            accel (float): target acceleration
        """
        if h_go < 0:
            h_go = self.h_go
        if h <= self.h_st:
            vh = 0
        elif self.h_st < h < h_go:
            vh = self.v_max / 2 * (1 - np.cos(np.pi * (h-self.h_st) / (h_go-self.h_st)))
            # vh = self.v_max * ((d-h_st) / (h_go-h_st))
        else:
            vh = self.v_max
        # alpha is applied to both headway based V and leading speed based V.
        return alpha*(vh-v) + beta*(v_lead-v)

