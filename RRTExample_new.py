import numpy as np
import matplotlib.pyplot as plt

from top_down import depth_to_xy_map
from RRTStarPlanner import RRTStarPlanner
from GlobalMapper import GlobalMapper

if __name__ == "__main__":
    K = np.array([[433.0, 0.0, 320.0],
                  [0.0, 433.0, 240.0],
                  [0.0, 0.0, 1.0]])
    
    # 1. Build map
    mapper = GlobalMapper(K, yaw_in_degrees=True, yaw_smoothing=0.8)
    for i in range(60):
        t = i * 0.25
        pose = {'north': t*0.8, 'east': t*0.1, 'yaw': 8.0 + i*0.015}
        # fake pose. Use the position monitor task 
        # #to get real pose in your implementation
        depth_img = np.zeros((480, 640), dtype=np.float32) 
        # fake depth image replace with your depth image grabbing code
        depth_img[100:400, 200:500] = 3.0 + np.random.rand()*0.05
        # fake depth image replace with your depth image grabbing code
        mapper.update_frame(depth_img, pose)
        
    obstacle_points = mapper.get_global_points()
    
    # 2. Plan path
    planner = RRTStarPlanner(
        safety_margin=0.6,  # Drone radius + buffer
        step_size=0.8,      # Smaller = smoother but slower
        max_iter=2000
    )
    
    start_pt = [0.0, 0.0]
    goal_pt  = [15.0, 2.0]
    path = planner.plan(start_pt, goal_pt, obstacle_points)
    
    # 3. Visualize
    fig, ax = plt.subplots(figsize=(7, 10))
    if len(obstacle_points) > 0:
        ax.scatter(obstacle_points[:, 1], obstacle_points[:, 0], 
                   s=3, c='gray', alpha=0.5, label='Obstacles')
    if path is not None:
        ax.plot(path[:, 1], path[:, 0], 'r-o', linewidth=2, markersize=4, label='Planned Path')
        ax.plot(start_pt[1], start_pt[0], 'go', markersize=12, label='Start')
        ax.plot(goal_pt[1], goal_pt[0], 'bx', markersize=14, label='Goal')
        
        # Show safety margin at waypoints
        for pt in path:
            circle = plt.Circle((pt[1], pt[0]), planner.safety_margin, 
                              color='r', fill=False, linestyle='--', alpha=0.3)
            ax.add_patch(circle)
            
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("RRT* Path on Global Point Cloud")
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.show()