import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=8, reward_type="shaped"):
        super(WarehouseEnv, self).__init__()
        self.grid_size = grid_size
        self.reward_type = reward_type
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4) 

        self.shelves = []
        for r in range(2, 6): 
            self.shelves.append((r, 2)); self.shelves.append((r, 5))
        self.charger_pos = (0, 0) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)

        self.battery = np.random.uniform(0.10, 1.0) 
        self.has_box = False
        self.charge_cooldown = False 
        
        self.robot_pos = self._get_empty_pos()
        while True:
            self.box_pos = self._get_empty_pos()
            self.target_pos = self._get_empty_pos()
            if self.box_pos != self.charger_pos and self.target_pos != self.charger_pos:
                dist = abs(self.box_pos[0] - self.target_pos[0]) + abs(self.box_pos[1] - self.target_pos[1])
                if dist > 4: break
        
        self.steps = 0
        self.last_target_type = self._get_target()[1]
        self.last_dist = self._get_distance()
        return self._get_obs(), {}

    def _get_empty_pos(self):
        while True:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if pos not in self.shelves and pos != (0,0): return pos

    def _get_target(self):
        if self.battery < 0.40: return self.charger_pos, "charger"
        if not self.has_box:    return self.box_pos, "box"
        return self.target_pos, "delivery"

    def _get_distance(self):
        target, _ = self._get_target()
        return abs(self.robot_pos[0] - target[0]) + abs(self.robot_pos[1] - target[1])

    def _get_obs(self):
        s = self.grid_size
        return np.array([
            self.robot_pos[0]/s, self.robot_pos[1]/s,
            self.box_pos[0]/s, self.box_pos[1]/s,
            self.target_pos[0]/s, self.target_pos[1]/s,
            1.0 if self.has_box else 0.0,
            self.battery 
        ], dtype=np.float32)

    def step(self, action):
        r, c = self.robot_pos
        old_pos = (r, c)
        
        if action == 0: r = max(0, r-1)
        elif action == 1: r = min(self.grid_size-1, r+1)
        elif action == 2: c = max(0, c-1)
        elif action == 3: c = min(self.grid_size-1, c+1)
        
        new_pos = (r, c)
        terminated = False
        self.steps += 1
        reward = -0.01 

        # 1. BATTERY
        drain = 0.005 if not self.has_box else 0.01
        self.battery -= drain

        # 2. CHARGING
        if new_pos == self.charger_pos:
            self.battery = 1.0
            # Only give reward if we haven't charged recently
            if not self.charge_cooldown:
                reward += 10.0
                self.charge_cooldown = True 
        
        # 3. MOVEMENT
        if new_pos in self.shelves:
            new_pos = old_pos
            reward += -0.5 
        self.robot_pos = new_pos

        # 4. DEATH
        if self.battery <= 0:
            reward += -100.0 
            terminated = True

        current_target_pos, current_target_type = self._get_target()
        dist = abs(self.robot_pos[0] - current_target_pos[0]) + abs(self.robot_pos[1] - current_target_pos[1])
        
        if current_target_type != self.last_target_type:
            self.last_dist = dist
            
        reward += (self.last_dist - dist) * 1.0 
        
        self.last_dist = dist
        self.last_target_type = current_target_type

        # 6. MILESTONES
        if not self.has_box and self.robot_pos == self.box_pos:
            self.has_box = True
            reward += 20.0 
            self.charge_cooldown = False # Reset charger reward capability

        if self.has_box and self.robot_pos == self.target_pos:
            reward += 100.0 
            terminated = True

        if self.steps >= 300: terminated = True
        
        return self._get_obs(), reward, terminated, False, {}