import gymnasium as gym
from gymnasium import spaces
import numpy as np




class RiverSwimEnv(gym.Env):
    def __init__(self,
                 normalize,
                 prob_stay_terminal=None):
        super().__init__()
        self.num_states = 6  # States: 0 to 5
        self.action_space = spaces.Discrete(2)  # Actions: 0 (swim left), 1 (swim right)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_states,), dtype=np.float32)  # One-hot states

        self.state = None

        self.t = None
        self.ep_reward = None
        self.horizon = 20
        self.normalize = normalize

        self.prob_stay = 0.6
        self.prob_right = 0.35
        self.prob_left = 0.05
        self.prob_stay_terminal = prob_stay_terminal if prob_stay_terminal is not None else self.prob_stay

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting state: Either 1 or 2 with equal probability
        self.state = np.random.choice([1, 2])
        self.t = 0
        self.ep_reward = 0
        return self._one_hot_encode(self.state), {}

    def step(self, action):
        reward = 0
        prev_state = self.state

        if action == 0:  # Swim left
            if self.state == 0:
                reward = 5  # Small reward for staying at state 0
                if self.normalize:
                    reward /= 10000
            else:
                self.state = max(0, self.state - 1)  # Move left

        elif action == 1:  # Swim right
            z = np.random.uniform()
            if self.state == 0:  # Initial state, higher probability to move right
                if z < 1 - self.prob_stay:
                    self.state = prev_state
                else:
                    self.state = self.state + 1
            elif self.state == self.num_states - 1:  # Terminal state with large reward
                if z < self.prob_stay_terminal:
                    self.state = prev_state
                    reward = 10000  # Large reward for staying at state 5
                    if self.normalize:
                        reward /= 10000
                else:
                    self.state = self.state - 1
            else:
                if z < self.prob_stay:  # Stay in the current state
                    self.state = prev_state
                elif self.prob_stay <= z < self.prob_stay + self.prob_left:  # Move left
                    self.state = max(0, self.state - 1)
                else:  # Move right
                    self.state = min(self.num_states - 1, self.state + 1)

        # Construct one-hot encoded state
        obs = self._one_hot_encode(self.state)
        self.t += 1
        self.ep_reward += reward

        done = (self.t == self.horizon)
        return obs, reward, done, False, {}

    def render(self):
        print(f"Current state: {self.state}")

    def close(self):
        pass

    def _one_hot_encode(self, state):
        """Return a one-hot encoded representation of the state."""
        one_hot = np.zeros(self.num_states, dtype=np.float32)
        one_hot[state] = 1.0
        return one_hot

    def compute_optVals(self):
        '''
        Compute optimal Q and V values of the environment by value iteation

        Args:
            NULL - works on the TabularMDP

        Returns:
            qVals - qVals[state, timestep] is vector of (optimal) Q values for each action
            vVals - vVals[timestep] is the vector of (optimal) V values at timestep
        '''
        qVals = {}
        vVals = {}

        vVals[self.horizon] = np.zeros(self.num_states)

        R_true = {}
        P_true = {}

        for s in range(self.num_states):
            for a in range(self.action_space.n):
                R_true[s, a] = (0, 0)
                P_true[s, a] = np.zeros(self.num_states)

        # Rewards
        R_true[0, 0] = (5. / 1000, 0)
        R_true[self.num_states - 1, 1] = (1, 0)

        # Transitions
        for s in range(self.num_states):
            P_true[s, 0][max(0, s - 1)] = 1.  # left action always succeed

        for s in range(1, self.num_states - 1):
            P_true[s, 1][min(self.num_states - 1, s + 1)] = 0.35
            P_true[s, 1][s] = 0.6
            P_true[s, 1][max(0, s - 1)] = 0.05

        P_true[0, 1][0] = 0.4
        P_true[0, 1][1] = 0.6
        P_true[self.num_states - 1, 1][self.num_states - 1] = 0.6
        P_true[self.num_states - 1, 1][self.num_states - 2] = 0.4

        for i in range(self.horizon):
            j = self.horizon - i - 1
            vVals[j] = np.zeros(self.num_states)

            for s in range(self.num_states):
                qVals[s, j] = np.zeros(self.action_space.n)

                for a in range(self.action_space.n):
                    qVals[s, j][a] = R_true[s, a][0] + np.dot(P_true[s, a], vVals[j + 1])

                vVals[j][s] = np.max(qVals[s, j])
        return qVals, vVals


# Register the environment with Gymnasium
if "RiverSwim-v0" not in list(gym.envs.registry.keys()):
    gym.envs.registration.register(
        id="RiverSwim-v0",
        entry_point="__main__:RiverSwimEnv",
    )

# Example Usage
if __name__ == "__main__":
    env = gym.make("RiverSwim-v0", normalize=True)
    obs, info = env.reset()
    print("Initial Observation:", obs)

    qVals, vVals = env.compute_optVals()

    for _ in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Observation: {obs}")
        env.render()