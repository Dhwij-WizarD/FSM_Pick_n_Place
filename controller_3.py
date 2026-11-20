import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# custom imports
from include.IK_3 import apply_ik_control
from include.utils_3 import get_blocking_objects, get_obj_pos, is_blocked, is_grasped, set_camera, safe_place, set_home, update_waypoint_markers, body_pos
from include.motion_planner_3 import MotionPlanner
from include.config_3 import (
    STEP_SIZE_FAST, STEP_SIZE_SLOW,
    GRIPPER_OPEN, GRIPPER_CLOSED,
    HOVER_DISTANCE, START_DELAY,
    APPROACH_THRESH, LOWER_THRESH, LIFT_THRESH, MOVE_THRESH, LOWER_PLACE_THRESH,
    WAIT_AFTER_LOWER, WAIT_AFTER_GRASP, WAIT_AFTER_RELEASE, WAIT_AFTER_DONE,
    EE_SITE_NAME, EE_SITE_ID_DEFAULT,LIFT_HEIGHT,obstacles
)

# .....─ Load Model ....................
model = mujoco.MjModel.from_xml_path('scene_3.xml')
data  = mujoco.MjData(model)

data.ctrl[3] = -1.25  # actuator4 initial position
data.ctrl[5] =  1.0  # actuator6 initial position
mujoco.mj_step(model, data)

# .....─ IK Setup ....................
target_rot = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()

try:
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
except Exception:
    ee_id = EE_SITE_ID_DEFAULT

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

# .....─ Motion Planner ....................
planner = MotionPlanner(lift_height=LIFT_HEIGHT, waypoint_thresh=0.03)

# .....─ State Variables ....................
current_state      = "HOLD"
gripper_cmd        = GRIPPER_OPEN
step_size          = STEP_SIZE_FAST

wait_counter       = 0
print_counter      = 0

target_pos         = None           # current IK target position
target_safe_pose   = safe_place()   # where to place the object
current_target_obj = None           # which object we are picking right now

exit_flag          = False
should_exit        = False
home_pose          = data.site_xpos[ee_id].copy()


task_queue = []

# .....─ Main Loop ....................
if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer)
        start_time = time.time()

        update_waypoint_markers(model, data, [])

        while viewer.is_running() and not should_exit:

            # ..... TASK PLANNING ....................
            if current_state == "HOLD":

                if time.time() - start_time > START_DELAY:
                    blockers = get_blocking_objects(model, data, obstacles)
                    if len(blockers) > 0:
                        for obj in blockers:
                            task_queue.append(("CLEAR", obj))
                            
                    task_queue.append(("PLACE", "red_box"))

                    if current_state == "HOLD" and len(task_queue) > 0:
                        task, obj = task_queue.pop(0)

                        if task == "CLEAR":
                            target_pos = body_pos(model, data, obj)
                            target_safe_pose = safe_place()
                            current_target_obj = obj

                        elif task == "PLACE":
                            target_pos = body_pos(model, data, "red_box")
                            target_safe_pose = body_pos(model, data, "dish") + np.array([0,0,HOVER_DISTANCE])
                            current_target_obj = "red_box"
                            exit_flag = True

                        planner.reset()
                        planner.plan(
                            start_pos=data.site_xpos[ee_id].copy(),
                            target_pos=target_pos.copy()
                        )
                        target_pos = planner.waypoints[0] if planner.waypoints else None
                        update_waypoint_markers(model, data, planner.waypoints)
                        current_state = "APPROACH"


            # ..... STATE MACHINE ....................
            if wait_counter > 1:
                wait_counter -= 1

            else:
                
                match current_state:

                    case "APPROACH":
                        # Follow motion planner waypoints to reach the object
                        step_size   = STEP_SIZE_FAST
                        planned_pos = planner.update(data.site_xpos[ee_id].copy())

                        if planned_pos is not None:
                            target_pos = planned_pos
                        else:
                            # All waypoints reached → go grasp
                            current_state = "GRASP"
                            wait_counter  = WAIT_AFTER_GRASP
                            update_waypoint_markers(model, data, [])  # hide markers
                            print("State: APPROACH → GRASP")

                        if print_counter % 100 == 0:
                            wp    = planner.current_index + 1
                            total = len(planner.waypoints)
                            print(f"State: APPROACH - Waypoint {wp}/{total}")


                    case "GRASP":
                        # Close gripper and verify grasp before proceeding
                        gripper_cmd = GRIPPER_CLOSED

                        if not is_grasped(model, data, obj_name=current_target_obj):
                            if print_counter % 10 == 0:
                                print("State: GRASP - Waiting for contact...")
                        else:
                            print(f"State: GRASP - {current_target_obj} grasped! → LIFT_UP")
                            current_state = "LIFT_UP"


                    case "LIFT_UP":
                            step_size   = STEP_SIZE_FAST
                            lift_target = data.site_xpos[ee_id].copy()
                            lift_target[2] += HOVER_DISTANCE
                            target_pos  = lift_target

                            error = np.linalg.norm(data.site_xpos[ee_id] - target_pos)
                            if error < LIFT_THRESH:
                                planner.plan(
                                    start_pos  = data.site_xpos[ee_id].copy(),
                                    target_pos = target_safe_pose.copy()
                                )
                                target_pos = planner.waypoints[0] if planner.waypoints else None
                                if len(planner.waypoints) == 0:   # ← correct Python list check
                                    print("State: LIFT_UP - No valid path to place location! Staying put.")
                                else:
                                    update_waypoint_markers(model, data, planner.waypoints)
                                    current_state = "MOVE_TO_PLACE"
                                    wait_counter  = WAIT_AFTER_GRASP   # ← delay so gripper settles before moving
                                    print("State: LIFT_UP → MOVE_TO_PLACE")

                            if print_counter % 100 == 0:
                                print(f"State: LIFT_UP - ERROR: {error:.4f}")


                    case "MOVE_TO_PLACE":
                        # Follow motion planner waypoints to place location
                        step_size   = STEP_SIZE_FAST
                        planned_pos = planner.update(data.site_xpos[ee_id].copy())

                        if planned_pos is not None:
                            target_pos = planned_pos
                        else:
                            # All waypoints reached → release
                            current_state = "RELEASE"
                            wait_counter  = WAIT_AFTER_RELEASE
                            update_waypoint_markers(model, data, [])  # hide markers
                            print("State: MOVE_TO_PLACE → RELEASE")

                        if print_counter % 100 == 0:
                            wp    = planner.current_index + 1
                            total = len(planner.waypoints)
                            print(f"State: MOVE_TO_PLACE - Waypoint {wp}/{total}")


                    case "RELEASE":
                        gripper_cmd = GRIPPER_OPEN

                        if wait_counter == 0:
                            wait_counter = 50   # wait ~50 simulation steps
                            print("State: RELEASE - Opening gripper...")

                        else:
                            # stay in same position while opening
                            target_pos = data.site_xpos[ee_id].copy()

                            if wait_counter == 1:
                                current_state = "DONE"
                                print("State: RELEASE → DONE")


                    case "DONE":
                        # Return to home pose, then go back to HOLD for next task
                        target_pos    = home_pose
                        wait_counter  = WAIT_AFTER_DONE
                        current_state = "HOLD"
                        planner.reset()
                        target_pos = planner.waypoints[0] if planner.waypoints else None  # set first target or None if no plan
                        update_waypoint_markers(model, data, [])  # hide markers
                        print("State: DONE - Returning home.")

                        if exit_flag :
                            print("All tasks complete. Exiting in 3 seconds...")
                            wait_counter = WAIT_AFTER_DONE + 50
                            current_state = "EXIT"

                    case "EXIT":
                        should_exit = True
                        print("Exiting now.")

            # ..... IK EXECUTION ....................
            if current_state != "HOLD" and target_pos is not None:
                apply_ik_control(
                    model, data,
                    target_p    = target_pos,
                    target_r    = target_rot,
                    gripper_val = gripper_cmd,
                    ee_id       = ee_id,
                    step_size   = step_size,
                    jacp        = jacp,
                    jacr        = jacr
                )

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
            print_counter += 1
