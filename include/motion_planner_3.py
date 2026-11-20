import numpy as np


class MotionPlanner:
    """
    Simple waypoint-based motion planner.
    Generates a safe sequence of 3 waypoints between current and target pose.

    Strategy (always):
        1. Lift straight up from current position    (avoid dragging objects)
        2. Move horizontally to above target         (avoid lateral collisions)
        3. Lower down to target                      (precise approach)
    """

    def __init__(self, lift_height=0.15, waypoint_thresh=0.03):
        """
        Args:
            lift_height     : how high to lift before moving horizontally (meters)
            waypoint_thresh : distance error to consider a waypoint reached (meters)
        """
        self.lift_height      = lift_height
        self.waypoint_thresh  = waypoint_thresh
        self.waypoints        = []
        self.current_index    = 0
        self.active           = False


    def plan(self, start_pos, target_pos):
        """
        Generate waypoints from start to target.

        Args:
            start_pos  : np.array (3,) — current ee position
            target_pos : np.array (3,) — desired ee position
        """
        lift_pos        = start_pos.copy()
        lift_pos[2]    += self.lift_height          # lift straight up

        above_target    = target_pos.copy()
        above_target[2] = lift_pos[2]               # same height, over target XY

        self.waypoints    = [
            lift_pos,       # waypoint 1: lift up
            above_target,   # waypoint 2: move horizontally
            target_pos,     # waypoint 3: lower to target
        ]
        self.current_index = 0
        self.active        = True

        print(f"[MotionPlanner] New plan: {len(self.waypoints)} waypoints.")
        for i, wp in enumerate(self.waypoints):
            print(f"  WP{i+1}: {np.round(wp, 3)}")


    def update(self, current_pos):
        """
        Call every simulation frame with the current ee position.
        Advances to the next waypoint when the current one is reached.

        Args:
            current_pos : np.array (3,) — current ee position
        Returns:
            current waypoint np.array(3,), or None if plan is complete.
        """
        if not self.active:
            return None

        waypoint = self._get_current_waypoint()
        if waypoint is None:
            return None

        error = np.linalg.norm(current_pos - waypoint)

        if error < self.waypoint_thresh:
            print(f"[MotionPlanner] Waypoint {self.current_index + 1}/{len(self.waypoints)} reached.")
            self.current_index += 1

            if self.current_index >= len(self.waypoints):
                self.active = False
                print("[MotionPlanner] Plan complete.")
                return None

        return self._get_current_waypoint()


    def reset(self):
        """Abort the current plan."""
        self.waypoints     = []
        self.current_index = 0
        self.active        = False
        print("[MotionPlanner] Plan reset.")


    def is_complete(self):
        """Returns True when all waypoints have been reached."""
        return not self.active and self.current_index >= len(self.waypoints)


    def _get_current_waypoint(self):
        if self.current_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_index]