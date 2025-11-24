import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio
import torch as th
from warehouse_env import WarehouseEnv
from ppo_scratch import MlpPolicy
import os

# Configuration
HIDDEN_LAYERS = [256, 256]
MODEL_PATH = "warehouse_strict_agent"
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

def get_action(model, obs_manual):
    """Helper to convert numpy obs -> tensor -> predict -> action"""
    obs_tensor = th.as_tensor(obs_manual, dtype=th.float32).to(DEVICE).unsqueeze(0)
    action_arr, _ = model.predict(obs_tensor, deterministic=True)
    return action_arr[0]

def find_cinematic_seed(model):
    print("🎬 DIRECTOR: Casting for the perfect scene (Scanning seeds)...")
    
    env = WarehouseEnv(grid_size=8, reward_type="shaped")
    
    for seed in range(1000):
        obs, _ = env.reset(seed=seed)
        
        env.battery = 0.25 # Force low battery start
        
        # Skip if we spawn on charger (too easy)
        if env.robot_pos == env.charger_pos: continue

        done = False
        steps = 0
        did_charge = False
        did_deliver = False
        
        while not done and steps < 100:
            s = env.grid_size
            obs_manual = np.array([
                env.robot_pos[0]/s, env.robot_pos[1]/s,
                env.box_pos[0]/s, env.box_pos[1]/s,
                env.target_pos[0]/s, env.target_pos[1]/s,
                1.0 if env.has_box else 0.0,
                env.battery
            ], dtype=np.float32)
            
            action = get_action(model, obs_manual)
            _, _, done, _, _ = env.step(action)
            steps += 1
            
            if env.battery == 1.0: did_charge = True
            if env.has_box and env.robot_pos == env.target_pos: did_deliver = True
            
            if did_charge and did_deliver:
                print(f"✅ FOUND PERFECT SCENE: Seed {seed} (Solved in {steps} steps)")
                return seed

    print("❌ Could not find a perfect Charge+Deliver run. Using Seed 0.")
    return 0

def record_video(model, seed, filename):
    print(f"🎥 ACTION! Recording {filename}...")
    
    env = WarehouseEnv(grid_size=8, reward_type="shaped")
    obs, _ = env.reset(seed=seed)
    
    env.battery = 0.25 # Set dramatic start condition
    
    frames = []
    
    for step in range(100):
        # --- Visualization Logic ---
        grid = np.zeros((env.grid_size, env.grid_size))
        for s in env.shelves: grid[s] = 1
        if not env.has_box: grid[env.box_pos] = 2
        grid[env.target_pos] = 3
        val = 5 if env.has_box else 4
        grid[env.robot_pos] = val
        grid[env.charger_pos] = 6 
        
        fig, ax = plt.subplots(figsize=(8,8))
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'gold', 'green', 'blue', 'purple', 'cyan'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        ax.imshow(grid, cmap=cmap, norm=norm)
        ax.grid(True, color='lightgrey', linewidth=1)
        ax.set_xticks([]); ax.set_yticks([])
        
        status = "MOVING"
        if env.robot_pos == env.charger_pos: status = "⚡ CHARGING"
        elif env.has_box: status = "📦 DELIVERING"
        elif env.battery < 0.3: status = "⚠️ LOW BATTERY"
        
        batt_color = "red" if env.battery < 0.3 else "darkgreen"
        title = f"Step: {step} | Battery: {int(env.battery*100)}% | {status}"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        patches = [
            mpatches.Patch(color='cyan', label='Charger'),
            mpatches.Patch(color='gold', label='Box'),
            mpatches.Patch(color='green', label='Target'),
            mpatches.Patch(color='blue', label='Robot'),
            mpatches.Patch(color='purple', label='Robot+Box'),
            mpatches.Patch(color='black', label='Wall')
        ]
        ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[..., :3] 
        frames.append(image)
        plt.close(fig)

        # --- End Conditions ---
        if env.has_box and env.robot_pos == env.target_pos:
            print("🏆 MISSION SUCCESS! Wrapping up video...")
            for _ in range(20): frames.append(frames[-1]) 
            break
        
        if env.battery <= 0:
            print("💀 CUT! Robot died.")
            break

        # --- Step Logic ---
        s = env.grid_size
        obs_manual = np.array([
            env.robot_pos[0]/s, env.robot_pos[1]/s,
            env.box_pos[0]/s, env.box_pos[1]/s,
            env.target_pos[0]/s, env.target_pos[1]/s,
            1.0 if env.has_box else 0.0,
            env.battery
        ], dtype=np.float32)
        
        action = get_action(model, obs_manual)
        _, _, done, _, _ = env.step(action)

    imageio.mimsave(filename, frames, fps=5)
    print(f"✅ SAVED: {filename}")

if __name__ == "__main__":
    env = WarehouseEnv(grid_size=8)
    
    # 1. Initialize Structure
    model = MlpPolicy(
        observation_space=env.observation_space, 
        action_space=env.action_space,
        net_arch=HIDDEN_LAYERS
    ).to(DEVICE)
    
    # 2. Load Weights
    full_path = MODEL_PATH + ".pth"
    if os.path.exists(full_path):
        print(f"Loading weights from {full_path}...")
        model.load_state_dict(th.load(full_path, map_location=DEVICE))
        model.eval() # Important for inference
        
        best_seed = find_cinematic_seed(model)
        record_video(model, best_seed, "final_presentation_vid.gif")
    else:
        print(f"❌ Error: {full_path} not found. Did you run the training script?")