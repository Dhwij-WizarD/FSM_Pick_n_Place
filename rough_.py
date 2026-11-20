import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Load the scene
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

if model.nkey > 0:
    data.qpos[:] = model.key_qpos[0]

# 2. Setup IK Parameters
alpha = 2.5  
target_rot = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()

ee_body_name = "hand" 
try:
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
except Exception as e:
    ee_id = 7 

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

def calculate_orientation_error(current_rot_mat, target_rot_mat):
    R_err = target_rot_mat @ current_rot_mat.T
    return R.from_matrix(R_err).as_rotvec()

def apply_ik_control(model, data, target_p, target_r, gripper_val,):
    """
    Applies IK and returns the position error norm so the state machine 
    knows when the robot has reached its target.
    """
    data.qfrc_applied[:model.nv] = data.qfrc_bias[:model.nv]

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data) 

    current_pos = data.xpos[ee_id]
    current_rot = data.xmat[ee_id].reshape(3, 3)
    
    pos_error = target_p - current_pos
    rot_error = calculate_orientation_error(current_rot, target_r)
    error = np.hstack([pos_error, rot_error])

    error = np.clip(error, -0.05, 0.05)

    mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
    J = np.vstack([jacp, jacr]) 

    delta_x = J.T @ error
    
    target_qpos_arm = data.qpos[:7] + (alpha * delta_x[:7])
    
    data.ctrl[:7] = target_qpos_arm

    if len(data.ctrl) > 7:
        data.ctrl[7:] = gripper_val
        
    return np.linalg.norm(pos_error)

# 3. Setup Pick and Place State Machine
GRIPPER_OPEN = 255  
GRIPPER_CLOSE = 0 

# Define Locations
pick_pos = np.array([0.5, 0.2, 0.06])
place_pos = np.array([0.4 + np.random.uniform(-0.1, 0.15), 
                      -0.2 + np.random.uniform(-0.15, 0.15), 
                      0.12])

# States
APPROACH = 0
LOWER = 1
GRASP = 2
LIFT_UP = 3
MOVE_TO_PLACE = 4
LOWER_PLACE = 5
RELEASE = 6
DONE = 7

current_state = APPROACH
current_target_pos = (pick_pos + np.array([0, 0, 0.7])) # Start 70cm above the cube
gripper_cmd = GRIPPER_OPEN
wait_counter = 0


def set_camera(viewer):
    viewer.cam.azimuth   = 180    # face from front
    viewer.cam.elevation = -30    # tilt down
    viewer.cam.distance  = 2.5    # zoom level
    viewer.cam.lookat[0] = 0.5    # x — center on table
    viewer.cam.lookat[1] = 0.0    # y
    viewer.cam.lookat[2] = 0.4    # z — table height


# 4. Launch the Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Simulation running. Target place pose generated at: {place_pos}")
    print("State: APPROACHing cube...")
    set_camera(viewer)  # <-- apply view immediately on launch
    input("Press Enter to start the pick and place task...")
    while viewer.is_running():
        step_start = time.time()

        # Step IK and get distance to target
        dist_to_target = apply_ik_control(model, data, current_target_pos, target_rot, gripper_cmd)

        if wait_counter > 0:
            wait_counter -= 1
        
        elif dist_to_target < 0.03: 
            if current_state == APPROACH:
                current_target_pos = pick_pos + np.array([0, 0, 0.15]) 
                current_state = LOWER
                alpha = 1.5 
                print("State: LOWERing to grasp...")
                
            elif current_state == LOWER:
                gripper_cmd = GRIPPER_CLOSE
                current_state = GRASP
                wait_counter = 60 
                print("State: GRASPing...")
                
            elif current_state == GRASP:
                current_target_pos = pick_pos + np.array([0, 0, 0.25])
                current_state = LIFT_UP
                print("State: LIFTing object...")
                
            elif current_state == LIFT_UP:
                current_target_pos = place_pos + np.array([0, 0, 0.25])
                current_state = MOVE_TO_PLACE
                print("State: MOVE to random place position...")
                
            elif current_state == MOVE_TO_PLACE:
                current_target_pos = place_pos
                current_state = LOWER_PLACE
                print("State: LOWERing to place...")
                
            elif current_state == LOWER_PLACE:
                gripper_cmd = GRIPPER_OPEN
                current_state = RELEASE
                wait_counter = 60 
                print("State: RELEASing object...")
                
            elif current_state == RELEASE:
                current_target_pos = place_pos + np.array([0, 0, 0.25])
                current_state = DONE
                print("Task Complete! Retreating arm...")

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)