import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


# IK Helper: Calculate orientation error as a rotation vector
def calculate_orientation_error(current_rot_mat, target_rot_mat):
    R_err = target_rot_mat @ current_rot_mat.T
    return R.from_matrix(R_err).as_rotvec()


# Main IK Control Function
def apply_ik_control(model, data, target_p, target_r, gripper_val, ee_id, step_size=1.5, jacp=None, jacr=None):

    data.qfrc_applied[:7] = data.qfrc_bias[:7]  # Compensate for gravity and Coriolis forces

    current_pos = data.site_xpos[ee_id]              # use site_xpos since ee_site is a site
    current_rot = data.site_xmat[ee_id].reshape(3, 3) # use site_xmat since ee_site is a site

    pos_error = target_p - current_pos
    rot_error = calculate_orientation_error(current_rot, target_r)
    error = np.hstack([pos_error, rot_error])

    error = np.clip(error, -0.05, 0.05)  # Limit error to prevent large jumps

    mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)  # use mj_jacSite since ee_site is a site
    J = np.vstack([jacp, jacr])

    if np.linalg.matrix_rank(J, tol=1e-6) < 6:
        print("Warning: Jacobian is rank deficient. IK solution may be unstable.")
        return np.linalg.norm(pos_error)
    
    delta_q = J.T @ error

    # To continuous steps (Don't know exactly why)
    if np.linalg.norm(delta_q) > 1e-6:
        delta_q = delta_q / np.linalg.norm(delta_q)

    target_qpos_arm = data.qpos[:7] + (step_size * delta_q[:7])
    data.ctrl[:7] = target_qpos_arm
    data.ctrl[7:] = gripper_val
