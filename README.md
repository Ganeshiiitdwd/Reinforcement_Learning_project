# 🤖 Autonomous Warehouse Agent with Battery Constraints using PPO

> **Course:** Reinforcement Learning / AI Games  
> **Project Type:** Resource-Constrained Navigation & Logistics

A comprehensive implementation of a **Proximal Policy Optimization (PPO)** agent designed to solve a multi-objective warehouse logistics problem. The agent must balance task completion (delivery) with survival (battery management) in a procedurally generated environment.

## 👨‍🎓 Group Information

| Name            | Roll Number | Branch |
| --------------- | ----------- | ------ |
| GANESH BHABAD   | 22BDS067    | DSAI   |
| AJAY RAJPUT     | 22BDS049    | DSAI   |
| ANIKET SHELKE   | 22BCS010    | CSE    |
| SHIVRAJ JAGDALE | 22BCS118    | CSE    |
| SHREYASH KADGE  | 22BCS120    | CSE    |

## 1. Problem Statement & Motivation

### The Challenge

The objective is to train an autonomous agent to operate within a grid-based **Warehouse Environment**. The agent is tasked with:

1.  **Navigation:** Avoiding static obstacles (shelves) to reach specific coordinates.
2.  **Logistics:** Picking up a package from a random location and delivering it to a target zone.
3.  **Survival (Critical):** Managing an internal battery level. If the battery depletes to 0, the agent "dies" (fails), incurring a massive penalty.

### 🌟 Why the "Battery Constrained" Version is Superior

Standard GridWorld or Maze navigation tasks are often trivialized by RL agents simply learning the shortest path. However, real-world autonomous mobile robots (AMRs) face **resource constraints**.

This constrained version provides significantly higher utility because:

1.  **Hierarchical Decision Making:** The agent cannot simply maximize the "Deliver" reward. It must dynamically switch its goal from "Deliver Box" to "Find Charger" based on its internal state (Battery Level).
2.  **Risk Management:** The agent learns to calculate if it has enough charge to reach the target or if it must detour to charge first.
3.  **Realism:** This closely mimics real-world warehouse scenarios where robot uptime and charging schedules are critical for throughput.
4.  **Reward Shaping Complexity:** It demonstrates how to balance dense shaping rewards (distance) with sparse terminal rewards (delivery) and survival penalties (death).

---

## 2. Codebase & Implementation Details

### A. Environment Logic & Neural Architecture

This section details how the physical world is translated into data (`warehouse_env.py`) and how the agent's "Brain" processes it (`ppo_scratch.py` / `stress_test_scratch.py`).

### 1. Observation Space (The Input)

The agent does not "see" the grid as an image (pixels). Instead, it perceives the world as a **Normalized Feature Vector of size 8**. This compact representation ensures faster training and generalization.

- **Type:** `Box(low=0, high=1, shape=(8,), dtype=float32)`
- **Normalization:** All coordinates are divided by `grid_size` (8) to keep values between $0.0$ and $1.0$.

| Index | Feature Name | Value Range    | Formula / Meaning                 |
| :---: | :----------- | :------------- | :-------------------------------- |
| **0** | Robot X      | $[0, 1]$       | $Pos_x / GridSize$                |
| **1** | Robot Y      | $[0, 1]$       | $Pos_y / GridSize$                |
| **2** | Box X        | $[0, 1]$       | $Box_x / GridSize$                |
| **3** | Box Y        | $[0, 1]$       | $Box_y / GridSize$                |
| **4** | Target X     | $[0, 1]$       | $Target_x / GridSize$             |
| **5** | Target Y     | $[0, 1]$       | $Target_y / GridSize$             |
| **6** | Has Box      | $\{0.0, 1.0\}$ | $1.0$ if carrying box, else $0.0$ |
| **7** | Battery      | $[0, 1]$       | Raw percentage ($1.0 = 100\%$)    |

### 2. Action Space (The Output)

The agent outputs a **Discrete Action** representing one of the four cardinal directions.

- **Type:** `Discrete(4)`

| Action Index | Command   | Description           |
| :----------: | :-------- | :-------------------- |
|    **0**     | **UP**    | Decrease Y coordinate |
|    **1**     | **DOWN**  | Increase Y coordinate |
|    **2**     | **LEFT**  | Decrease X coordinate |
|    **3**     | **RIGHT** | Increase X coordinate |

### 3. Neural Network Architecture (Actor-Critic)

We utilize an **Actor-Critic** architecture. This means the Neural Network has two distinct "Heads" (outputs) that perform different tasks, often sharing the initial processing layers.

**Visual Architecture Diagram:**

```text
       [ INPUT LAYER ]
     (8 normalized floats)
             |
             v
    [ HIDDEN LAYERS (MLP) ]
    (e.g., 256 neurons, Tanh)
             |
             v
    +--------+--------+
    |                 |
    v                 v
[ ACTOR HEAD ]   [ CRITIC HEAD ]
    |                 |
 (Softmax)         (Linear)
    |                 |
    v                 v
[ ACTION PROBS ]   [ STATE VALUE ]
 (4 probabilities)    (1 Scalar)
```

#### Explanation of Outputs

**A. The Actor Head (Policy $\pi_\theta$)**

- **Role:** The "Doer." It decides what action to take.
- **Output:** A vector of 4 probabilities (summing to $1.0$).
- **Formula:** It uses the **Softmax** function to convert raw network logits ($z$) into probabilities:
  $$\pi(a_i|s) = \frac{e^{z_i}}{\sum_{j=0}^{3} e^{z_j}}$$
- **Selection:**
  - _Training:_ We **sample** from this distribution (enables exploration).
  - _Testing:_ We take the **argmax** (highest probability).

**B. The Critic Head (Value Function $V_\phi$)**

- **Role:** The "Coach." It estimates how good the current state is.
- **Output:** A single scalar number (e.g., $15.5$ or $-2.0$).
- **Meaning:** It predicts the **Discounted Future Return** (total reward the agent expects to get from this point until the end of the episode).
- **Formula:**
  $$V(s) \approx \mathbb{E} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$
- **Why it's needed:** The Critic's prediction is compared against the actual reward received to calculate the **Advantage** (did we do better or worse than expected?), which is used to train the Actor.

**Code Snippet:**

```python
# From warehouse_env.py

def _get_obs(self):
    s = self.grid_size
    return np.array([
        self.robot_pos[0] / s, self.robot_pos[1] / s,   # Robot Coordinates
        self.box_pos[0] / s, self.box_pos[1] / s,       # Box Coordinates
        self.target_pos[0] / s, self.target_pos[1] / s, # Target Coordinates
        1.0 if self.has_box else 0.0,                   # State Flag
        self.battery                                    # Critical Resource
    ], dtype=np.float32)
```

#### Key Feature: Reward Shaping

To prevent sparse reward issues, we implemented specific shaping logic. The step function calculates rewards based on distance changes and battery status.

```python
# From warehouse_env.py

# 1. BATTERY DRAIN
drain = 0.005 if not self.has_box else 0.01 # Drain faster if carrying load
self.battery -= drain

# 2. CHARGING LOGIC
if new_pos == self.charger_pos:
    self.battery = 1.0
    if not self.charge_cooldown:
        reward += 10.0 # Incentive to charge, but only if needed

# 3. TERMINAL PENALTY
if self.battery <= 0:
    reward += -100.0 # Massive penalty for death
    terminated = True
```

_Explanation:_ The code explicitly penalizes carrying a box to simulate weight/physics. The charging logic includes a cooldown to prevent the agent from "camping" on the charger to farm infinite rewards.

### B. The Algorithm: `ppo_scratch.py`

This file contains the "from scratch" implementation of PPO, demonstrating the mathematical mechanics of the algorithm.

#### 1. Generalized Advantage Estimation (GAE)

We implement GAE inside `RolloutBuffer` to reduce variance in trajectory estimation.

**Mathematical Formula:**
The advantage $\hat{A}_t$ is calculated recursively:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$$

**Visual Logic Flow (Backward Pass):**

```text
Step T (End) --> Step T-1 --> ... --> Step 0
      ^               ^                 ^
      |               |                 |
  [Calc Delta]    [Add Future Adv]  [Result stored]
```

**Code Snippet:**

```python
# From ppo_scratch.py
# We loop BACKWARDS (reversed) because GAE relies on the "next" step's advantage
for step in reversed(range(self.buffer_size)):
    # Delta term represents the TD-Error (Temporal Difference)
    delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]

    # Recursive formula: Current Delta + Discounted Future Advantage
    last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
    self.advantages[step] = last_gae_lam
```

#### 2. The Actor-Critic Network (`MlpPolicy`)

We use a shared backbone network that splits into two heads.

**Architecture Diagram:**

```text
State Input (8)
      |
[ Shared Linear Layer 64 ]
      |
      +---------------------+
      |                     |
[ Actor Head ]        [ Critic Head ]
      |                     |
   Softmax               Linear
      |                     |
Action Probs (4)      Value Estimate (1)
```

**Code Snippet:**

```python
# From ppo_scratch.py
self.pi_net = nn.Sequential(*pi_layers) # Actor Network
self.vf_net = nn.Sequential(*vf_layers) # Critic Network

if self.is_discrete:
    logits = self.action_head(pi_latent)
    # Categorical distribution allows us to sample discrete actions (0,1,2,3)
    dist = Categorical(logits=logits)
```

#### 3. The PPO Update Loop

The core learning mechanism uses the **Clipped Surrogate Objective**. This forces the update to stay within a safe range, preventing "catastrophic forgetting."

**Mathematical Formula:**

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

Where $r_t(\theta)$ is the probability ratio $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

**Visual Logic (Clipping):**

```text
   Ratio          |
      Is ratio > 1.2 ?
      /           \
   YES             NO     |
    |               |
[Clip to 1.2] [Keep Ratio]     |
    |               |
    +-----+------+
          |
   Multiply by Advantage
          |
    Update Gradient
```

**Code Snippet:**

```python
# From ppo_scratch.py

# Ratio indicates how much more likely the action is NOW vs. when we collected data
ratio = th.exp(log_prob - old_log_probs)

# Unclipped objective: Standard Policy Gradient
policy_loss_1 = advs * ratio

# Clipped objective: Forces ratio to stay between 0.8 and 1.2 (if epsilon=0.2)
policy_loss_2 = advs * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

# We take the minimum (pessimistic bound) to be safe
policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
```

### C. Testing Suite: `stress_test_analytics.py`

This script runs a rigorous evaluation of the trained model, focusing on edge cases that standard training metrics might miss.

#### Critical Battery Initialization

To test "Intelligence," we force the robot to start with low battery.

```python
# From stress_test_analytics.py
start_batt = np.random.uniform(0.15, 1.0) # Randomize battery
env.battery = start_batt

# Define "Critical" as less than 30% charge
is_critical = start_batt < 0.30

# Metric tracking logic:
if is_critical:
    if did_charge: stats["critical_saves"] += 1 # Intelligent behavior
    else: stats["critical_fails"] += 1          # Failed to prioritize survival
```

---

## 3. Results & Performance Analysis

### A. Training Metrics (Graphs)

We monitored the training process over **2,000,000 timesteps**.

<p style="text-align: center;">
  <img src="graph2.jpeg" width="600" title="GRAPH 2">
</p>

**Graph 1: Episode Reward (`rollout/ep_rew_mean`)**

- **Trend:** The curve begins at a negative value (due to the -100 death penalty and wandering).
- **Inflection Point:** Around 250k steps, the reward spikes, indicating the agent has learned to avoid obstacles and find the target.
- **Convergence:** The reward stabilizes around **143.9**, which represents a near-perfect run (100 Delivery + 20 Pickup + Distance Bonuses - minimal Battery Costs).

<p style="text-align: center;">
  <img src="graph1.jpeg" width="600" title="GRAPH 1">
</p>

**Graph 2: Episode Length (`rollout/ep_len_mean`)**

- **Trend:** Starts high (~200 steps) as the agent explores randomly.
- **Convergence:** Drops to **~20 steps** per episode.
- **Interpretation:** This confirms the agent is taking the **shortest path** possible. It is no longer wandering; it is executing the mission with high efficiency.

### B. Stress Test Results (Quantitative Table)

The model `warehouse_strict_agent` was subjected to 1,000 randomized test episodes.

| Metric           | Value   | Interpretation                                                |
| :--------------- | :------ | :------------------------------------------------------------ |
| **Duration**     | 38.06 s | The model is lightweight and highly performant.               |
| **Success Rate** | 84.20%  | The agent successfully delivers the package most of the time. |
| **Avg Steps**    | 23.3    | Matches the theoretical optimal path length for an 8×8 grid.  |

#### Failure Mode Analysis

- **Battery Deaths (8.5%):** Occurs in "Impossible Scenarios" where the agent spawns with 15% battery but the charger is on the other side of the map (mathematically impossible to reach).
- **Timeouts (7.3%):** Rare instances where the agent enters a loop.

#### 🧠 Intelligence Check (The "Why it's Useful" Metric)

This specific metric determines if the agent learned **Priority Management**.

- **Scenario:** Battery starts below 30% (Critical).
- **Total Critical Episodes:** 160
- **Survived & Delivered:** 145
- **Dead:** 15
- **Survival Rate:** **90.6%**

### C. Visual Demonstration

Here is the trained agent in action.

- **Dark Blue Square:** Robot
- **Yellow Suare:** Box
- **Green Square:** Target Delivery Zone
- **Cyan Square:** Charging Station

<p align="center">
  <img src="final_presentation_video-looped_inf.gif" width="600" alt="Agent Demo GIF">
  <br>
  <em>Figure 3: The agent successfully identifies low battery (Red Bar), navigates to charge, picks up the package, and delivers it.</em>
</p>

**Conclusion:** The agent successfully learned to prioritize survival over task completion when resources are low. It demonstrates **emergent intelligent behavior**.

---

## 4. How to Run

### Prerequisites

Ensure you have the required libraries installed:

```bash
pip install torch numpy gymnasium stable-baselines3 tensorboard tqdm matplotlib imageio
```

### 1. Training the Agent

To train the agent using the robust PPO implementation (SB3 wrapper):

```bash
python train_compare.py
```

- **Output:** Generates `warehouse_strict_agent.zip` and TensorBoard logs in `./logs/strict/`.

### 2. Monitoring Training

To visualize the graphs shown in the report:

```bash
tensorboard --logdir ./logs/
```

### 3. Running Stress Tests

To generate the detailed statistics table:

```bash
python stress_test_analytics.py
```

### 4. Visual Demo (GIF Generation)

To create a video file (`final_presentation_video.gif`) showing the agent in action:

```bash
python create_demo.py
```

---

## 5. Conclusion

This project successfully implements a **Battery-Constrained Autonomous Agent** using Proximal Policy Optimization. By integrating resource constraints into the standard warehouse logistics problem, we created an agent capable of **dynamic decision-making**. The results—specifically the **90.6% survival rate** in critical battery states—demonstrate that Deep Reinforcement Learning is a viable solution for complex, multi-objective robotic control systems.
