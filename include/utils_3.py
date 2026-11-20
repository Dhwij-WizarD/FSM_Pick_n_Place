import mujoco
import numpy as np


def set_camera(viewer):
    viewer.cam.azimuth   = 180    # face from front
    viewer.cam.elevation = -30    # tilt down
    viewer.cam.distance  = 2.0    # zoom level
    viewer.cam.lookat[0] = 0.5    # x — center on table
    viewer.cam.lookat[1] = 0.0    # y
    viewer.cam.lookat[2] = 0.4    # z — table height


def get_obj_pos(name, model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id != -1:
        return data.xpos[body_id]

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id != -1:
        return data.site_xpos[site_id]

    raise ValueError(f"Object '{name}' not found as body or site in model.")


def is_blocked(model, data, item_1="green_box", item_2="dish"):
    pos1 = get_obj_pos(item_1, model, data)
    pos2 = get_obj_pos(item_2, model, data)
    dist = np.linalg.norm(pos1 - pos2)
    return dist < 0.15  # 15cm threshold


def is_grasped(model, data, obj_name, finger1_name="left_finger", finger2_name="right_finger"):
    """
    Check if the object is being grasped by verifying both fingers
    are in contact with the object simultaneously.

    Args:
        model        : MuJoCo model
        data         : MuJoCo data
        obj_name     : name of the object body to check (e.g. "green_box")
        finger1_name : name of left finger body
        finger2_name : name of right finger body
    Returns:
        True if both fingers are contacting the object, False otherwise.
    """
    obj_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, finger1_name)
    finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, finger2_name)

    finger1_contact = False
    finger2_contact = False

    for i in range(data.ncon):
        contact    = data.contact[i]
        geom1_body = model.geom_bodyid[contact.geom1]
        geom2_body = model.geom_bodyid[contact.geom2]
        bodies     = {geom1_body, geom2_body}

        if obj_id in bodies:
            if finger1_id in bodies:
                finger1_contact = True
            if finger2_id in bodies:
                finger2_contact = True

    return finger1_contact and finger2_contact


def update_waypoint_markers(model, data, waypoints):
    """
    Move wp1/wp2/wp3 sites to the current plan's waypoints.
    Uses model.site_pos (writable) not data.site_xpos (read-only).

    Call with planner.waypoints to show markers.
    Call with [] to hide all markers below the floor.
    """
    HIDDEN       = np.array([0.0, 0.0, -1.0])
    marker_names = ["wp1", "wp2", "wp3"]

    for i, name in enumerate(marker_names):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id == -1:
            print(f"[Markers] Warning: site '{name}' not found in model.")
            continue

        if i < len(waypoints):
            model.site_pos[site_id] = waypoints[i]   # ← model not data
        else:
            model.site_pos[site_id] = HIDDEN          # ← model not data


def safe_place():
    """Uniform random placement around dish."""
    center = np.array([0.3, 1.0, 0.4])

    xy = center[:2] + np.random.uniform(-0.1, 0.1, size=2)
    z  = center[2] + 0.1

    xy = np.clip(xy, [0.3, -0.3], [0.7, 0.3])

    return np.array([xy[0], xy[1], z])


def set_home(data):
    data.qpos[1] = 0.0
    data.qpos[2] = 0.0
    data.qpos[3] = -1.2
    data.qpos[4] = 0.0
    data.qpos[5] = 1.0
    data.qpos[6] = 0.0


def body_pos(model, data,name):
    """Shorthand to get a body position as a copy."""
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def get_blocking_objects(model, data, obstacles, dish_name="dish"):
    blockers = []
    dish_pos = body_pos(model, data, dish_name)

    for obj in obstacles:
        if obj == "red_box":
            continue

        obj_pos = body_pos(model, data, obj)
        dist = np.linalg.norm(obj_pos - dish_pos)

        if dist < 0.15:  # same threshold
            blockers.append(obj)

        if is_on_top(model, data, obj, "red_box"): # if obj is on top of red_box, also consider it a blocker
            blockers.append(obj)

    return blockers

def is_on_top(model, data, obj1, obj2):
    pos1 = body_pos(model, data, obj1)
    pos2 = body_pos(model, data, obj2)

    return (
        abs(pos1[0] - pos2[0]) < 0.05 and
        abs(pos1[1] - pos2[1]) < 0.05 and
        pos1[2] > pos2[2]
    )