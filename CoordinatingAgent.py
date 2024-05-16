import numpy as np
import math

from SimpleTagAgent import SimpleTagAgent

class CoordinatingAgent(SimpleTagAgent):

    def __init__(self, name, num_adversaries, num_landmarks):
        super().__init__(name, num_adversaries, num_landmarks)
        self.update_target_divisor_threshold = 1
        self.target_vec_divisor = 1
        self.target_vec_multiplier = 1.3
        self.target_vec_divisor_upper_limit = 3

    # Functions
    def get_action(self):
        target_vecs = self.get_target_vecs()
        agent_abs_pos = self.observed_agent_positions["agent_0"]
        target_points = np.add(target_vecs, agent_abs_pos)

        dists_from_target_points = {}
        for ag in self.agents:
            if not ag == "agent_0":
                pos = self.observed_agent_positions[ag]
                diff_vects = np.subtract(target_points, pos)
                dists = np.linalg.norm(diff_vects, axis=1)
                dists_from_target_points[ag] = dists

        assigned_targets = self.assign_target_points_to_adversaries(self.agents, dists_from_target_points) 

        target_rel_pos = target_points[assigned_targets[self.name]] - self.observed_agent_positions[self.name] 
        max_action = self.get_best_action_wrt_target_point(self.action_vectors, target_rel_pos)

        total_dist_from_target = self.get_total_dist_from_target_points(assigned_targets, target_points)
        if total_dist_from_target < self.update_target_divisor_threshold:
            self.update_target_vec_divisor()
        return max_action
    
    def deg_to_rad(self, degrees):
        radians = degrees * (math.pi / 180)
        return radians
    
    def unit_vector_from_radians(self, radians):
        x = math.cos(radians) if abs(math.cos(radians)) > 1e-5 else 0
        y = math.sin(radians) if abs(math.sin(radians)) > 1e-5 else 0
        length = math.sqrt(x**2 + y**2)
        return x/length, y/length

    def get_target_vecs(self):
        # 360 degrees in a circle
        deg = 360 / self.num_adversaries
        rad = self.deg_to_rad(deg)
        target_vecs = []
        for i in range(self.num_adversaries):
            vect_x, vect_y = self.unit_vector_from_radians(rad*i)
            target_vecs.append((vect_x / self.target_vec_divisor, vect_y / self.target_vec_divisor))
        return target_vecs
    
    def get_best_action_wrt_target_point(self, action_vectors, target_rel_pos):
        max_dist = 0
        max_action = None
        for action, action_vect in action_vectors.items():
            dist = np.dot(action_vect, target_rel_pos)
            if dist > max_dist:
                max_dist = dist
                max_action = action
        return max_action
    
    def get_total_dist_from_target_points(self, assigned_targets, target_points):
        tot_dist = 0
        for agent in self.agents:
            if agent != "agent_0":
                pos = self.observed_agent_positions[agent]
                tot_dist += np.linalg.norm(target_points[assigned_targets[agent]] - pos)
        return tot_dist

    def assign_target_points_to_adversaries(self, agents, dists_from_target_points):
        num_assigned = 0
        assigned_targets = {agent: False for agent in agents}
        done = False
        while not done:
            min_agent = None
            min_dist = math.inf
            min_dist_index = None
            for a, dists in dists_from_target_points.items():
                if min(dists) < min_dist:
                    min_dist = min(dists)
                    min_agent = a
                    min_dist_index = np.argmin(dists)
            assigned_targets[min_agent] = min_dist_index
            del dists_from_target_points[min_agent]
            for v in dists_from_target_points.values():
                v[min_dist_index] = math.inf
            num_assigned += 1
            if num_assigned == self.num_adversaries:
                done = True
        return assigned_targets

    def update_target_vec_divisor(self):
        self.target_vec_divisor = min(self.target_vec_multiplier * self.target_vec_divisor, self.target_vec_divisor_upper_limit)
        print(f"Updated target vec divisor: {self.target_vec_divisor}")

