import torch as th
import numpy as np
from warehouse_env import WarehouseEnv
from ppo_scratch import MlpPolicy  # Importing from your custom PPO file
import os
import time

# Configuration to match training
HIDDEN_LAYERS = [256, 256] 
MODEL_PATH = "warehouse_strict_agent"

def load_policy(env, device):
    """Recreates the architecture and loads weights from the .pth file"""
    policy = MlpPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=HIDDEN_LAYERS
    ).to(device)
    
    full_path = MODEL_PATH + ".pth"
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Could not find model file: {full_path}")
        
    print(f"Loading weights from {full_path}...")
    # Load state dict
    policy.load_state_dict(th.load(full_path, map_location=device))
    policy.eval() # Set to evaluation mode
    return policy

def run_stress_test(num_episodes=1000):
    print(f"📊 STARTING SCRATCH PPO STRESS TEST: {num_episodes} Randomized Episodes")
    print(f"   Testing robust navigation, battery management, and pathfinding...")
    print("-" * 65)

    env = WarehouseEnv(grid_size=8, reward_type="shaped")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Load the custom model
    try:
        policy = load_policy(env, device)
    except FileNotFoundError as e:
        print(e)
        return None, 0

    # Metrics
    stats = {
        "total": 0,
        "success": 0,
        "died": 0,
        "timeout": 0,
        "steps_taken": [],
        "critical_saves": 0,   
        "critical_fails": 0,   
        "normal_wins": 0       
    }

    start_time = time.time()

    for i in range(num_episodes):
        obs, _ = env.reset(seed=i) 
        
        # Inject specific test scenarios
        start_batt = np.random.uniform(0.15, 1.0)
        env.battery = start_batt
        
        is_critical = start_batt < 0.30
        did_charge = False
        done = False
        
        while not done:
            # 1. Prepare Observation for Custom PPO (Numpy -> Tensor -> Batch Dim)
            # Your MlpPolicy expects a torch tensor
            obs_tensor = th.as_tensor(obs, dtype=th.float32).to(device)
            
            # Add batch dimension: shape becomes (1, 8)
            obs_tensor = obs_tensor.unsqueeze(0)
            
            # 2. Predict
            # Your custom predict returns (action, state)
            action_arr, _ = policy.predict(obs_tensor, deterministic=True)
            
            # Extract scalar action from the array (since batch size is 1)
            action = action_arr[0] 

            # 3. Step
            obs, _, done, _, _ = env.step(action)
            
            # --- Tracking Logic (Same as before) ---
            if env.robot_pos == env.charger_pos and env.battery == 1.0:
                did_charge = True

            if env.has_box and env.robot_pos == env.target_pos:
                stats["success"] += 1
                stats["steps_taken"].append(env.steps)
                
                if is_critical:
                    if did_charge: stats["critical_saves"] += 1
                else:
                    stats["normal_wins"] += 1
                break
            
            if env.battery <= 0:
                stats["died"] += 1
                if is_critical: stats["critical_fails"] += 1
                break
            
            if env.steps >= 200:
                stats["timeout"] += 1
                break

    stats["total"] = num_episodes
    duration = time.time() - start_time
    
    return stats, duration

def print_report(stats, duration):
    if stats is None: return

    success_rate = (stats["success"] / stats["total"]) * 100
    avg_steps = np.mean(stats["steps_taken"]) if stats["steps_taken"] else 0
    
    print("\n" + "="*65)
    print(f"🧪 STRESS TEST RESULTS (Duration: {duration:.2f}s)")
    print("="*65)
    print(f"🏆 OVERALL SUCCESS RATE:  {success_rate:.2f}%")
    print(f"📉 Average Steps to Win:  {avg_steps:.1f}")
    print("-" * 65)
    print(f"❌ FAILURE MODES:")
    print(f"   💀 Battery Deaths:    {stats['died']} ({stats['died']/stats['total']*100:.1f}%)")
    print(f"   ⌛ Timeouts (Stuck):  {stats['timeout']} ({stats['timeout']/stats['total']*100:.1f}%)")
    print("-" * 65)
    print(f"🧠 INTELLIGENCE CHECK:")
    
    total_crit = stats['critical_saves'] + stats['critical_fails']
    crit_rate = (stats['critical_saves'] / total_crit * 100) if total_crit > 0 else 0
    
    print(f"   🔋 Critical Start Handling (<30% Batt):")
    print(f"      - Survived & Delivered: {stats['critical_saves']}")
    print(f"      - Died:                 {stats['critical_fails']}")
    print(f"      - Survival Rate:        {crit_rate:.1f}%")
    
    print("="*65)
    
    if success_rate > 90:
        print("✅ VERDICT: PRODUCTION READY (Grade A)")
    elif success_rate > 75:
        print("⚠️ VERDICT: ACCEPTABLE (Grade B)")
    else:
        print("❌ VERDICT: UNSTABLE (Grade F)")

if __name__ == "__main__":
    stats, duration = run_stress_test(num_episodes=1000)
    print_report(stats, duration)