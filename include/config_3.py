# ─── IK Parameters ───────────────────────────────────────────
STEP_SIZE_FAST          = 0.05   # used during APPROACH, LIFT_UP, MOVE_TO_PLACE
STEP_SIZE_SLOW          = 0.03   # used during LOWER, LOWER_PLACE

# ─── Gripper ─────────────────────────────────────────────────
GRIPPER_OPEN            = 255
GRIPPER_CLOSED          = 0.0

# ─── Task Parameters ─────────────────────────────────────────
HOVER_DISTANCE          = 0.1    # how high above object to hover (meters)
START_DELAY             = 3      # seconds to wait before starting task

# ─── Error Thresholds (when to transition states) ────────────
APPROACH_THRESH         = 0.05
LOWER_THRESH            = 0.01
LIFT_THRESH             = 1.005
MOVE_THRESH             = 0.05
LOWER_PLACE_THRESH      = 0.03

# ─── Wait Counters (simulation steps) ────────────────────────
WAIT_AFTER_LOWER        = 5
WAIT_AFTER_GRASP        = 20
WAIT_AFTER_RELEASE      = 50
WAIT_AFTER_DONE         = 100

# ─── End Effector ────────────────────────────────────────────
EE_SITE_NAME            = "ee_site"
EE_SITE_ID_DEFAULT      = 8

# ─── Planner  ────────────────────────────────────────────
LIFT_HEIGHT            = 0.1   # how high to lift before moving horizontally (meters)
obstacles              = ["red_box", "green_box1", "green_box2", "green_box3"]  # names of objects to avoid when planning