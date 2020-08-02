import numpy as np
from physics_sim import PhysicsSim

# Get to the height of 50

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Tell if the episode finished successfully
        self.is_successfull = False

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([
                                                                             0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # distance between current and target positions
        # we take 100 as this is bigger than the target height of 50
        # and so this is safe to compare with calculated position.
        # original: reward = 1.-.3*(abs(self.sim.pose[:2] - self.target_pos[2])).sum()
        return 1 - min(self.target_pos[2] - self.sim.pose[2], 100)/100

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        done = False
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        # set the final goal
        if done:
            # reward or penalize
            if self.sim.pose[2] >= self.target_pos[2]:
                # mark this episode as successful
                self.is_successfull = True
                reward = 5
            else:
                reward = -5

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.is_successfull = False
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
