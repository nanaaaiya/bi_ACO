import yaml
from PIL import Image
import numpy as np
import math
import heapq
import sys # error handling
import matplotlib.pyplot as plt 
import csv

# max penalty for unreachable paths
MAX_PENALTY = 1000000.0 

INSTANCE_SEED = 80552497 
rng_inst = np.random.RandomState(INSTANCE_SEED)


def load_map(yaml_file):
    try:
        with open(yaml_file, "r") as f:
            info = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: YAML file '{yaml_file}' not found.")
        sys.exit(1)

    try:
        img_path = info["image"]
        img = Image.open(img_path).convert('L') # convert to grayscale if not already
    except FileNotFoundError:
        print(f"Error: PGM image file '{info['image']}' not found.")
        sys.exit(1)

    grid = np.array(img).astype(np.uint8)
    grid = np.flipud(grid)  # flip y-axis 
    # grid = np.fliplr(grid)  # flip x-axis

    # convert 1 = obstacle, 0 = free
    occ = np.where(grid < 128, 1, 0) 

    resolution = info["resolution"]
    origin = info["origin"]

    return occ, resolution, origin, img_path 

def world_to_grid(x, y, origin, res):
    gx = int((x - origin[0]) / res)
    gy = int((y - origin[1]) / res)
    return (gx, gy)

def grid_to_world(gx, gy, origin, res):
    x = gx * res + origin[0] + res / 2
    y = gy * res + origin[1] + res / 2
    return (x, y)

def sample_valid_position(rng, occ_grid, origin, res, x_min, x_max, y_min, y_max):
    rows, cols = occ_grid.shape
    attempts = 0
    max_attempts = 1000 
    while attempts < max_attempts:
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        
        gx, gy = world_to_grid(x, y, origin, res)
        
        if 0 <= gx < cols and 0 <= gy < rows:
            if occ_grid[gy][gx] == 0:
                return x, y
        attempts += 1
    
    raise RuntimeError(f"Could not find a valid free space position after {max_attempts} attempts. Check map bounds/coverage.")


def plot_everything(occ, robots, tasks, drops, paths_world, origin, res, title=None, save_filename=None, draw_paths=True):
    plt.figure(figsize=(10,10))
    plt.imshow(occ, cmap="gray_r") 

    robots_g = [world_to_grid(x,y,origin,res) for (x,y) in robots]
    tasks_g = [world_to_grid(x,y,origin,res) for (x,y) in tasks]
    drops_g = [world_to_grid(x,y,origin,res) for (x,y) in drops]

    robot_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
    
    if paths_world:
        robot_task_assignments = {} 
        robot_dropoff_assignments = {}  
        
        sorted_paths = sorted(paths_world.items())
        for robot_idx, (name, pts) in enumerate(sorted_paths):
            robot_task_assignments[robot_idx] = set()
            robot_dropoff_assignments[robot_idx] = set()
            
            for pt in pts[1:]:  
                pt_xy = (pt[0], pt[1]) 
                
                matched_task = False
                for task_idx, task_pos in enumerate(tasks):
                    if np.isclose(pt_xy[0], task_pos[0], atol=0.01) and np.isclose(pt_xy[1], task_pos[1], atol=0.01):
                        robot_task_assignments[robot_idx].add(task_idx)
                        matched_task = True
                        break
                
                if not matched_task:
                    for dropoff_idx, dropoff_pos in enumerate(drops):
                        if np.isclose(pt_xy[0], dropoff_pos[0], atol=0.01) and np.isclose(pt_xy[1], dropoff_pos[1], atol=0.01):
                            robot_dropoff_assignments[robot_idx].add(dropoff_idx)
                            break
    else:
        robot_task_assignments = {}
        robot_dropoff_assignments = {}

    # robots (red)
    for i,(gx,gy) in enumerate(robots_g):
        robot_color = 'red'
        plt.plot(gx, gy, "o", color=robot_color, markersize=8)
        plt.text(gx, gy + 3, f"R{i}", color=robot_color, fontsize=16)

    # tasks
    for j,(gx,gy) in enumerate(tasks_g):
        task_color = 'red'
        if paths_world:
            for robot_idx, task_set in robot_task_assignments.items():
                if j in task_set:
                    task_color = robot_colors[robot_idx % len(robot_colors)]
                    break
        plt.plot(gx, gy, "o", color=task_color, markersize=8)
        plt.text(gx, gy + 3, f"T{j}", color=task_color, fontsize=16)

    # dropoffs
    for k,(gx,gy) in enumerate(drops_g):
        dropoff_color = 'green'
        if paths_world:
            for robot_idx, dropoff_set in robot_dropoff_assignments.items():
                if k in dropoff_set:
                    dropoff_color = robot_colors[robot_idx % len(robot_colors)]
                    break
        plt.plot(gx, gy, "o", color=dropoff_color, markersize=8)
        plt.text(gx, gy + 3, f"D{k}", color=dropoff_color, fontsize=16)

    if draw_paths:
        sorted_paths = sorted(paths_world.items()) if paths_world else []
        for robot_idx, (name, pts) in enumerate(sorted_paths):
            if not pts or len(pts) < 2:
                continue
            
            path_color = robot_colors[robot_idx % len(robot_colors)]
            
            full_path_g = []
            pts_g = [world_to_grid(x,y,origin,res) for (x,y) in pts]
            
            for i in range(len(pts_g) - 1):
                start_g = pts_g[i]
                goal_g = pts_g[i + 1]
                
                segment_path = astar_path(occ, start_g, goal_g)
                if segment_path:
                    if full_path_g:
                        full_path_g.extend(segment_path[1:])
                    else:
                        full_path_g.extend(segment_path)
            
            if full_path_g:
                xs = [p[0] for p in full_path_g]
                ys = [p[1] for p in full_path_g]
                plt.plot(xs, ys, linewidth=2, linestyle='--', alpha=0.7, color=path_color)

    if title:
        plt.title(title, fontsize=18)
    else:
        plt.title("Map, Robots, Tasks, Paths, and World Origin", fontsize=18)
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
        plt.close()
    else:
        plt.show()

def astar_full(grid, start, goal):
    rows, cols = grid.shape
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

    def h(a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    open_set = []
    heapq.heappush(open_set, (0 + h(start, goal), 0, start))
    came_from = {start: None}
    g = {start: 0}

    while open_set:
        f, cost, node = heapq.heappop(open_set)
        if node == goal:
            return None, g[goal] 

        x, y = node
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= cols or ny >= rows:
                continue
            if grid[ny][nx] == 1: 
                continue

            step_cost = math.hypot(dx, dy) 
            new_g = g[node] + step_cost
            nxt = (nx, ny)

            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                came_from[nxt] = node
                f = new_g + h(nxt, goal)
                heapq.heappush(open_set, (f, new_g, nxt))

    return None, float("inf")

def astar_path(grid, start, goal):
    rows, cols = grid.shape
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

    def h(a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    open_set = []
    heapq.heappush(open_set, (0 + h(start, goal), 0, start))
    came_from = {start: None}
    g = {start: 0}

    while open_set:
        f, cost, node = heapq.heappop(open_set)
        if node == goal:
            path = []
            current = goal
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        x, y = node
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= cols or ny >= rows:
                continue
            if grid[ny][nx] == 1:
                continue

            step_cost = math.hypot(dx, dy) 
            new_g = g[node] + step_cost
            nxt = (nx, ny)

            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                came_from[nxt] = node
                f = new_g + h(nxt, goal)
                heapq.heappush(open_set, (f, new_g, nxt))

    return None  


def compute_astar_distance_matrices(occ, origin, res, robot_pos, task_pos, drop_pos):
    nR = len(robot_pos)
    mT = len(task_pos)
    mD = len(drop_pos) 

    robot_g = [world_to_grid(x, y, origin, res) for (x, y) in robot_pos]
    task_g = [world_to_grid(x, y, origin, res) for (x, y) in task_pos]
    drop_g = [world_to_grid(x, y, origin, res) for (x, y) in drop_pos]

    dist_robot_to_task = np.full((nR, mT), float('inf'))
    dist_task_to_drop = np.full(mT, float('inf')) 
    dist_drop_to_task = np.full((mD, mT), float('inf'))
    dist_task_to_task = np.full((mT, mT), float('inf'))

    print("Computing A* distances...")

    def get_dist(start_g, goal_g):
        _, cost = astar_full(occ, start_g, goal_g)
        return cost * res

    # 1. R -> T
    for i in range(nR):
        for j in range(mT):
            dist_robot_to_task[i, j] = get_dist(robot_g[i], task_g[j])

    # 2. T -> D (pickup -> dropoff)
    for t in range(mT):
        dist_task_to_drop[t] = get_dist(task_g[t], drop_g[t])

    # 3. D -> T (dropoff -> next pickup)
    for d in range(mD):
        for t in range(mT):
            dist_drop_to_task[d, t] = get_dist(drop_g[d], task_g[t])

    # 4. T -> T
    for j in range(mT):
        for k in range(mT):
            if j == k:
                dist_task_to_task[j, k] = 0.0
                continue
            dist_task_to_task[j, k] = get_dist(task_g[j], task_g[k])
            
    print("A* distance computation complete.")

    is_inf = np.isinf(dist_robot_to_task).any() or np.isinf(dist_task_to_drop).any() or np.isinf(dist_drop_to_task).any() or np.isinf(dist_task_to_task).any()

    if is_inf:
        print(f"Warning: Some paths are unreachable. Replacing 'inf' with large penalty cost: {MAX_PENALTY:.2f}")

    dist_robot_to_task[np.isinf(dist_robot_to_task)] = MAX_PENALTY
    dist_task_to_drop[np.isinf(dist_task_to_drop)] = MAX_PENALTY
    dist_drop_to_task[np.isinf(dist_drop_to_task)] = MAX_PENALTY
    dist_task_to_task[np.isinf(dist_task_to_task)] = MAX_PENALTY
    # ------------------------------------------------------------------------

    return dist_robot_to_task, dist_task_to_drop, dist_drop_to_task, dist_task_to_task


occ_grid, res, origin, _ = load_map("swarm_arena_map.yaml")
rows, cols = occ_grid.shape
print(f"Map Loaded. Dimensions: {rows} rows x {cols} cols. Resolution: {res} m/cell")

# define problem size
n_robots = 3
m_tasks = 20

# robot start positions
robot_positions_world = np.array([
    (0.0, -3.0), # tb1
    (3.85, -1.0), # tb2
    (-3.5, 1.35) # tb3
])

# task positions
task_x_min = origin[0] + 0.5; task_x_max = origin[0] + 15.0
task_y_min = origin[1] + 0.5; task_y_max = origin[1] + 13.0

task_positions_world = np.zeros((m_tasks, 2))
for i in range(m_tasks):
    x, y = sample_valid_position(rng_inst, occ_grid, origin, res, 
                                 task_x_min, task_x_max, task_y_min, task_y_max)
    task_positions_world[i] = (x, y)

m_tasks = len(task_positions_world)

# dropoff positions
dropoff_positions_world = np.zeros((m_tasks, 2))
offset_range = 2.0 # within +/- 2.0m of the task center
for i in range(m_tasks):
    tx, ty = task_positions_world[i]
    
    d_x_min = tx - offset_range; d_x_max = tx + offset_range
    d_y_min = ty - offset_range; d_y_max = ty + offset_range
    
    x, y = sample_valid_position(rng_inst, occ_grid, origin, res, 
                                 d_x_min, d_x_max, d_y_min, d_y_max)
    dropoff_positions_world[i] = (x, y)

(dist_robot_to_task, 
 dist_task_to_drop, 
 dist_drop_to_task, 
 dist_task_to_task) = compute_astar_distance_matrices(occ_grid, origin, res, 
                                                    robot_positions_world, 
                                                    task_positions_world, 
                                                    dropoff_positions_world)
                                                            
robot_labels = [f"tb{i+1}" for i in range(n_robots)]

def build_waypoints_dict(routes):
    WAYPOINTS = {}
    for i in range(n_robots):
        path = []
        seq = routes.get(i, [])
        
        start_x, start_y = robot_positions_world[i]
        gx_r, gy_r = world_to_grid(start_x, start_y, origin, res)
        wx_r, wy_r = grid_to_world(gx_r, gy_r, origin, res)
        path.append( (round(float(wx_r),3), round(float(wy_r),3), 0.0) )
        
        for idx in seq:
            if idx < m_tasks:
                x,y = task_positions_world[idx]
            else:
                x,y = dropoff_positions_world[idx - m_tasks]
            
            gx, gy = world_to_grid(x, y, origin, res)
            wx, wy = grid_to_world(gx, gy, origin, res)
            path.append( (round(float(wx),3), round(float(wy),3), 0.0) )
    
        WAYPOINTS[robot_labels[i]] = path
    return WAYPOINTS

def run_aco(run_seed, num_ants=30, num_iterations=100, alpha=1.0, beta=2.0, rho=0.1, q=100.0, verbose=False, history_out=None):
    rng = np.random.RandomState(int(run_seed))

    pher_assign = np.ones((m_tasks, n_robots))
    pher_seq = np.ones((m_tasks, m_tasks))
    heuristic_assign = 1.0 / (dist_robot_to_task.T + 1e-6)
    heuristic_seq = 1.0 / (dist_task_to_task + np.eye(m_tasks))
    heuristic_assign[heuristic_assign < 1e-10] = 1e-10

    best_cost = float('inf')
    best_routes = None
    random_ant_routes = None
    random_ant_cost = None

    for iteration in range(num_iterations):
        solutions = []
        for ant_idx in range(num_ants):
            assignment = np.zeros(m_tasks, dtype=int)
            for t in range(m_tasks):
                weights = (pher_assign[t] ** alpha) * (heuristic_assign[t] ** beta)
                weights /= weights.sum() + 1e-12
                assignment[t] = rng.choice(n_robots, p=weights)

            robot_tasks = {i: [] for i in range(n_robots)}
            for t in range(m_tasks):
                robot_tasks[assignment[t]].append(t)

            for i in range(n_robots):
                tasks = robot_tasks[i]
                if len(tasks) <= 1:
                    continue

                sequence = []
                remaining = set(tasks)
                current_task_idx = None

                if remaining:
                    start_task = min(remaining, key=lambda t: dist_robot_to_task[i, t])
                    sequence.append(start_task)
                    remaining.remove(start_task)
                    current_task_idx = start_task

                while remaining:
                    candidates = list(remaining)
                    probs = []
                    for next_task in candidates:
                        pher_val = pher_seq[current_task_idx, next_task] ** alpha
                        cost_dt = dist_drop_to_task[current_task_idx, next_task]
                        heur_val = 1.0 / (cost_dt + 1e-6) ** beta
                        probs.append(pher_val * heur_val)
                    probs = np.array(probs)
                    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)
                    idx = rng.choice(len(candidates), p=probs)
                    next_task = candidates[idx]

                    sequence.append(next_task)
                    remaining.remove(next_task)
                    current_task_idx = next_task

                robot_tasks[i] = sequence

            routes = {}
            total_cost = 0.0
            for i in range(n_robots):
                rt = []
                dist_sum = 0.0
                seq = robot_tasks[i]
                for idx, t in enumerate(seq):
                    rt.append(t)
                    rt.append(t + m_tasks)
                    if idx == 0:
                        dist_sum += dist_robot_to_task[i, t]
                    dist_sum += dist_task_to_drop[t]
                    if idx < len(seq) - 1:
                        dist_sum += dist_drop_to_task[t, seq[idx + 1]]
                routes[i] = rt
                total_cost += dist_sum

            solutions.append((assignment.copy(), robot_tasks, total_cost, routes))
            if total_cost < best_cost:
                best_cost = total_cost
                best_routes = routes.copy()

        if solutions:
            iter_best = min(s[2] for s in solutions)
            if verbose:
                print(
                    f"[ACO] Iter {iteration+1}/{num_iterations}: "
                    f"iter_best_total_distance={iter_best:.2f} "
                    f"best_so_far_total_distance={best_cost:.2f}"
                )
            if history_out is not None:
                history_out.append(
                    {
                        "algorithm": "ACO",
                        "iteration": int(iteration + 1),
                        "iter_best_total_distance": float(iter_best),
                        "best_so_far_total_distance": float(best_cost),
                        "unique_costs": None,
                    }
                )

        if iteration == 0 and random_ant_routes is None:
            # select random ant from first iteration
            random_ant_idx = rng.randint(0, len(solutions))
            random_ant_solution = solutions[random_ant_idx]
            random_ant_cost = random_ant_solution[2]  # total_cost
            random_ant_routes = random_ant_solution[3].copy()  # routes

        # pheromone evaporation
        pher_assign *= (1.0 - rho)
        pher_seq *= (1.0 - rho)
        for assignment, robot_tasks_seq, cost, routes in solutions:
            deposit = q / (cost + 1e-9)
            for t, r in enumerate(assignment):
                pher_assign[t, r] += deposit
            for i in range(n_robots):
                seq = robot_tasks_seq[i]
                for k in range(len(seq)-1):
                    a, b = seq[k], seq[k+1]
                    pher_seq[a, b] += deposit

    waypoints = build_waypoints_dict(best_routes)
    total = best_cost
    
    random_ant_waypoints = None
    if random_ant_routes is not None:
        random_ant_waypoints = build_waypoints_dict(random_ant_routes)
    
    return waypoints, total, best_routes, random_ant_waypoints, random_ant_routes


if __name__ == "__main__":
    aco_history = []
    aco_waypoints, aco_total, aco_routes, aco_random_ant_waypoints, aco_random_ant_routes = run_aco(INSTANCE_SEED, verbose=True, history_out=aco_history)
    print("ACO WAYPOINTS = {")
    for k,v in aco_waypoints.items():
        print(f"    \"{k}\": {v},")
    print("}")
    print("Total Distance: {:.2f}".format(aco_total))

    # Save PSO + ACO distances over iterations to CSV
    csv_path = "distances_over_iterations.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "iteration",
                "iter_best_total_distance",
                "best_so_far_total_distance",
                "unique_costs",
            ],
        )
        writer.writeheader()

        for row in aco_history:
            writer.writerow(row)
    print(f"[CSV] Saved iteration distances to: {csv_path}")
    

    aco_paths_world = {}
    for i in range(n_robots):
        path = []
        route = aco_routes.get(i, [])
        
        path.append(robot_positions_world[i])
        
        for idx in route:
            if idx < m_tasks:
                path.append(tuple(task_positions_world[idx]))
            else:
                path.append(tuple(dropoff_positions_world[idx - m_tasks]))
        
        aco_paths_world[robot_labels[i]] = path
    
    random_ant_paths_world = {}
    random_ant_total_distance = 0.0
    
    if aco_random_ant_routes is not None:
        for i in range(n_robots):
            route = aco_random_ant_routes.get(i, [])
            path = []
            dist_sum = 0.0
            path.append(robot_positions_world[i])
            prev_task_idx = None
            for j in range(0, len(route), 2):
                if j + 1 < len(route):
                    task_idx = route[j]
                    dropoff_idx = route[j + 1] - m_tasks  
                    
                    path.append(tuple(task_positions_world[task_idx]))
                    path.append(tuple(dropoff_positions_world[dropoff_idx]))

                    if prev_task_idx is None:
                        dist_sum += dist_robot_to_task[i, task_idx]
                    else:
                        dist_sum += dist_drop_to_task[prev_task_idx, task_idx]
                    
                    dist_sum += dist_task_to_drop[task_idx]
                    
                    prev_task_idx = task_idx
                elif j < len(route):
                    task_idx = route[j]
                    path.append(tuple(task_positions_world[task_idx]))
                    if prev_task_idx is None:
                        dist_sum += dist_robot_to_task[i, task_idx]
                    else:
                        dist_sum += dist_drop_to_task[prev_task_idx, task_idx]
                    dist_sum += dist_task_to_drop[task_idx]
            
            random_ant_paths_world[robot_labels[i]] = path
            random_ant_total_distance += dist_sum
    else:
        random_ant_paths_world = aco_paths_world.copy()
        random_ant_total_distance = aco_total
    
    plot_everything(
        occ_grid,
        robot_positions_world,
        task_positions_world,
        dropoff_positions_world,
        random_ant_paths_world,  
        origin,
        res,
        title=f"Initial Map Setup ({n_robots} Robots, {m_tasks} Tasks) - Random Ant Initial Solution (Total Distance: {random_ant_total_distance:.2f}m)",
        save_filename="figure1_initial_setup.png",
        draw_paths=True  
    )
    
    plot_everything(
        occ_grid,
        robot_positions_world,
        task_positions_world,
        dropoff_positions_world,
        aco_paths_world,
        origin,
        res,
        title=f"ACO Optimized Paths (Total Distance: {aco_total:.2f}m)",
        save_filename="figure2_aco_optimized.png"
    )