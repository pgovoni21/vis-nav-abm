"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""

import pygame
import numpy as np
from abm.contrib import colors, decision_params, movement_params
from abm.agent import supcalc
from collections import OrderedDict
import importlib

import pprint
pp = pprint.PrettyPrinter(depth=4)
import time

class Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad, pooling_time,
                 pooling_prob, consumption, vision_range, visual_exclusion, patchwise_exclusion):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent in env as (x, y)
        :param orientation: absolute orientation of the agent (0: right, pi/2: up, pi: left, 3*pi/2: down)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param v_field_res: resolution of the visual field of the agent in pixels
        :param FOV: visual field as a tuple of min max visible angles e.g. (-np.pi, np.pi)
        :param window_pad: padding of the environment in simulation window in pixels
        :param pooling_time: time units needed to pool status of a given position in the environment
        :param pooling_prob: initial probability to switch to pooling behavior
        :param consumption: (resource unit/time unit) consumption efficiency of agent
        :param vision_range: in px the range/radius in which the agent is able to see other agents
        :param visual_exclusion: if True social cues can be visually excluded by non social cues.
        :param patchwise_exclusion: exclude agents from visual field if exploiting the same patch
        :param behave_params: dictionary of behavioral parameters can be passed to a given agent which will
            overwrite the parameters defined in the env files. (e.g. when agents are heterogeneous)
        """
        
### -------------------------- INITIALIZATION -------------------------- ###

        # PyGame Sprite superclass
        super().__init__()

        # in case we run multiple simulations, we reload the env parameters
        importlib.reload(decision_params)
        importlib.reload(movement_params)

        # Saved parameters
        self.id = id 
        self.position = np.array(position, dtype=np.float64) ### dynamic sensory input
        self.orientation = orientation ### dynamic sensory input

        self.velocity = 0  # (absolute) ### dynamic sensory input?
        self.mode = "explore"  # explore, flock, collide, exploit, pool
        
        self.exploited_patch_id = -1 
        self.collected_r = 0  # resource units collected by agent 
        self.collected_r_before = 0  # ^ from previous time step to monitor patch quality (not saved)
        
        # Visual parameters
        self.v_field_res = v_field_res
        self.FOV = FOV
        self.vision_range = vision_range
        self.visual_exclusion = visual_exclusion

        self.soc_v_field = np.zeros(self.v_field_res)  # social visual projection field
        self.vis_field_source_data = {} # to calculate relevant visual field according to relocation force

        # Behavioral parameters
        self.exp_stop_ratio = movement_params.exp_stop_ratio
        # self.max_exp_vel = movement_params.exp_vel_max ## not in humanexp8
        self.consumption = consumption
        self.exclude_agents_same_patch = patchwise_exclusion
        self.overriding_mode = None
        ## w
        self.S_wu = decision_params.S_wu
        self.T_w = decision_params.T_w
        self.w = 0
        self.Eps_w = decision_params.Eps_w
        self.g_w = decision_params.g_w
        self.B_w = decision_params.B_w
        self.w_max = decision_params.w_max
        ## u
        self.I_priv = 0  # saved
        self.novelty = np.zeros(decision_params.Tau)
        self.S_uw = decision_params.S_uw
        self.T_u = decision_params.T_u
        self.u = 0
        self.Eps_u = decision_params.Eps_u
        self.g_u = decision_params.g_u
        self.B_u = decision_params.B_u
        self.u_max = decision_params.u_max
        self.F_N = decision_params.F_N
        self.F_R = decision_params.F_R

        # Pooling attributes
        self.pooling_time = pooling_time
        self.pooling_prob = pooling_prob
        
        self.time_spent_pooling = 0  # time units currently spent with pooling the status of given position (changes dynamically)
        self.env_status_before = 0
        self.env_status = 0  # status of the environment in current position, 1 if rescource, 0 otherwise
        self.pool_success = 0  # states if the agent deserves 1 piece of update about the status of env in given pos

        # Environment related parameters
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.window_pad = window_pad
        self.boundaries_x = [self.window_pad, self.window_pad + self.WIDTH]
        self.boundaries_y = [self.window_pad, self.window_pad + self.HEIGHT]

        # Visualization / human interaction parameters
        self.radius = radius
        self.color = color
        self.selected_color = colors.LIGHT_BLUE

        self.show_stats = False
        self.is_moved_with_cursor = 0

        # Initial Visualization of agent
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(self.image, color, (radius, radius), radius)
        # Showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, colors.BACKGROUND, (radius, radius),
                         ((1 + np.cos(self.orientation)) * radius, (1 - np.sin(self.orientation)) * radius), 3)
        self.rect = self.image.get_rect()
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]
        self.mask = pygame.mask.from_surface(self.image)

### -------------------------- VISUAL PROJECTION FUNCTIONS -------------------------- ###

    def calc_social_V_proj(self, agents):
        """
        Calculates socially relevant visual projection field of the agent. This is calculated as the
        projection of nearby exploiting agents that are not visually excluded by other agents
        """
        # gather all visible agents (exluding self)
        agents_vis = [ag for ag in agents if supcalc.distance(self, ag) <= self.vision_range and ag.id != self.id]
        
        # separate agents that are exploiting on non-emptied patch with the others
        agents_vis_expl = []
        agents_vis_nonexpl = []
        for ag in agents_vis:
            if ag.get_mode() == "exploit" and ag.exploited_patch_id != self.exploited_patch_id: ## assuming patchwise_social_exclusion == True
                agents_vis_expl.append(ag) 
            else: # if ag.get_mode() != "exploit" or ag.exploited_patch_id == self.exploited_patch_id
                agents_vis_nonexpl.append(ag)

        # calculate visual field either regarding other agents as obstacles or not
        if self.visual_exclusion:
            self.soc_v_field = self.projection_field(agents_vis_expl, keep_distance_info=False,
                                                     non_expl_agents = agents_vis_nonexpl)
        else:
            self.soc_v_field = self.projection_field(agents_vis_expl, keep_distance_info=False)
    
    def projection_field(self, obstacles, keep_distance_info=False, non_expl_agents=None):
        """
        Calculates visual projection field for the agent given the visible obstacles in the environment
        :param obstacles: list of agents (with same radius) or some other obstacle sprites to generate projection field
        :param keep_distance_info: if True, the amplitude of the vpf will reflect the distance of the object from the
            agent so that exclusion can be easily generated with a single computational step.
        :param non_expl_agents: a list of non-scoial visual cues (non-exploiting agents) that on the other hand can still
            produce visual exlusion on the projection of social cues. If None only social cues can produce visual
            exclusion on each other.
        :param fov: agent field of view as percentage, e.g. 0.5 corresponds to a visual range of [-pi/2, pi/2]
        """

        # extracting obstacle coordinates
        obstacle_coords = [ob.position for ob in obstacles]

        # if non-social cues can visually exclude social ones we also concatenate these to the obstacle coords
        if non_expl_agents is not None:
            len_social = len(obstacles) # to demarcate this divide when iterating in list below
            obstacle_coords.extend([ob.position for ob in non_expl_agents])

        # initializing visual field and relative angles at the specified resolution
        v_field = np.zeros(self.v_field_res)
        phis = np.linspace(-np.pi*self.FOV, np.pi*self.FOV, self.v_field_res)

        # center point of self
        pt_self_c = self.position + self.radius

        # edge point on agent's perimeter according to its orientation
        pt_self_e = np.array([
            pt_self_c[0] + np.cos(self.orientation) * self.radius, 
            pt_self_c[1] - np.sin(self.orientation) * self.radius])

        # direction vector, magnitude = radius, flipped y-axis
        vec_self_dir = pt_self_e - pt_self_c
        ## where v1[0] --> + : right, 0 : center, - : left, 10 : max
        ## where v1[1] --> + : down, 0 : center, - : up, 10 : max

        # calculating orientation angle between agent/obstacle + visual exclusionary angle +  projection size
        self.vis_field_source_data = {}
        for i, obstacle_coord in enumerate(obstacle_coords):

            # center of obstacle (other agent)
            pt_other_c = obstacle_coord + self.radius

            # vector between obstacle center + self edge, magnitude = distance
            vec_between = pt_other_c - pt_self_e

            # calculating orientation angle with respect to direction vector of self + vector bw agent-obstacle
            # requires the magnitude/norm of both vectors (radius + distance)
            distance = np.linalg.norm(vec_between)
            angle_between = supcalc.angle_between(vec_self_dir, vec_between, self.radius, distance)
            ## relative to perceiving agent, in radians between [-pi (left/CCW), +pi (right/CW)]

            # print("agent id/pos/orient/dir: ", self.id, self.position, self.orientation*180/np.pi, vec_self_dir)
            # print("between angle/vector: ", angle_between*180/np.pi, vec_between)

            # if target is visible, we calculate projection + save relevant data into dict
            if abs(angle_between) <= phis[-1]: # self.FOV*np.pi or positive limit of visible range

                # finding where in the retina the projection belongs to
                phi_target = supcalc.find_nearest(phis, angle_between) # where in retina the obstacle is

                # calculating visual exclusionary angle between obstacle + self
                # assumes obstacle = agent radius (change for landmarks)
                angle_vis_excl = 2 * np.arctan(self.radius / distance)

                # calculating projection size / limits
                proj_prop_total = angle_vis_excl / (2 * np.pi) # exclusion as proportion of entire 360 deg 2D plane
                proj_prop_FOV = proj_prop_total / self.FOV # as proportion of the field of view
                proj_size = proj_prop_FOV * self.v_field_res # discretized to visual resolution units
                proj_L = int(phi_target - proj_size / 2) # CCW from center
                proj_R = int(phi_target + proj_size / 2) # CW from center

                # imposing boundary conditions to projection field
                if self.FOV < 1: # no-slip
                    if proj_L < 0: 
                        proj_L = 0
                    if proj_R > self.v_field_res: 
                        proj_R = self.v_field_res
                else: # self.FOV = 1
                    print("Error - periodic boundary conditions not applied")
                    pass # to be finished later, if at all, biologically implausible + complicates exclusion calculation
                    # if proj_L < 0:
                    #     v_field[self.v_field_res + proj_L:self.v_field_res] = 1
                    #     proj_L = 0
                    # if proj_R >= self.v_field_res:
                    #     v_field[0:proj_R - self.v_field_res] = 1
                    #     proj_R = self.v_field_res - 1

                # saving relevant data to dict
                self.vis_field_source_data[i] = {}
                self.vis_field_source_data[i]["angle_vis_excl"] = angle_vis_excl
                self.vis_field_source_data[i]["phi_target"] = phi_target
                self.vis_field_source_data[i]["distance"] = distance

                self.vis_field_source_data[i]["proj_L"] = proj_L
                self.vis_field_source_data[i]["proj_R"] = proj_R

                self.vis_field_source_data[i]["proj_L_ex"] = proj_L
                self.vis_field_source_data[i]["proj_R_ex"] = proj_R

                # marking whether observed agents were exploiting or not
                if non_expl_agents is not None: # non-exploiting agents exist + are in the visual projection field
                    if i < len_social: # exploiting agents
                        self.vis_field_source_data[i]["is_social_cue"] = True
                    else: # non-exploiting agents
                        self.vis_field_source_data[i]["is_social_cue"] = False
                else: # no non-expoiting agents exist and/or are observed
                    self.vis_field_source_data[i]["is_social_cue"] = True

        # calculating visual exclusion if requested
        if self.visual_exclusion and len(self.vis_field_source_data.items()) > 1:
            self.exclude_V_source_data()

        # removing non-exploiting agents (non-social cues)
        if non_expl_agents is not None:
            self.remove_nonsocial_V_source_data()

        # sorting VPF source data according to visual angle
        self.rank_V_source_data("angle_vis_excl")

        # if len(self.vis_field_source_data) > 1:
        #     pp.pprint(self.vis_field_source_data)

        # marking exploiting agents in visual field
        for k, v in self.vis_field_source_data.items():
            angle_vis_excl = v["angle_vis_excl"]
            phi_target = v["phi_target"]
            proj_L = v["proj_L_ex"]
            proj_R = v["proj_R_ex"]

            # weighing projection amplitude with rank information if requested
            if not keep_distance_info: # always false --> agents cannot interpret distance
                v_field[proj_L:proj_R] = 1
            else:
                v_field[proj_L:proj_R] = (1 - distance / self.vision_range)

        return v_field

    def exclude_V_source_data(self):
        """
        Iterates over pairs of obstacles, excluding occluded obstacles/parts
        """
        # ranks obstacles according to distance - low first
        self.rank_V_source_data("distance", reverse=False)

        # combination operation, though itertools doesn't seem to provide performance benefit and is less understandable
        for _, obs_close in self.vis_field_source_data.items(): 
            for _, obs_far in list(self.vis_field_source_data.items())[1:]: 
                if obs_far["distance"] > obs_close["distance"]: 
                    # Partial R-side exclusion
                    if obs_far["proj_R_ex"] > obs_close["proj_L"] > obs_far["proj_L_ex"]: 
                        obs_far["proj_R_ex"] = obs_close["proj_L"]
                        continue
                    # Partial L-side exclusion
                    if obs_far["proj_L_ex"] < obs_close["proj_R"] < obs_far["proj_R_ex"]:
                        obs_far["proj_L_ex"] = obs_close["proj_R"]
                        continue
                    # Total exclusion
                    if obs_close["proj_L"] <= obs_far["proj_L_ex"] and obs_close["proj_R"] >= obs_far["proj_R_ex"]:
                        obs_far["proj_L_ex"] = 0
                        obs_far["proj_R_ex"] = 0
    
    def rank_V_source_data(self, ranking_key, reverse=True):
        """
        Ranks/sorts visual projection field data by specified key
        :param: ranking_key: string
        """
        sorted_dict = sorted(self.vis_field_source_data.items(), key=lambda kv: kv[1][ranking_key], reverse=reverse)
        self.vis_field_source_data = OrderedDict(sorted_dict)

    def remove_nonsocial_V_source_data(self):
        """
        Creates new data with purely social/exploiting agent data after calculating exclusions + before interaction processes
        """
        clean_sdata = {}
        for k, v in self.vis_field_source_data.items():
            if v['is_social_cue']: # == True
                clean_sdata[k] = v
        self.vis_field_source_data = clean_sdata

### -------------------------- DECISION MAKING FUNCTIONS -------------------------- ###

    def calc_I_priv(self):
        """returning I_priv according to the environment status. Note that this is not necessarily the same as
        later on I_priv also includes the reward amount in the last n timesteps"""
        # other part is coming from uncovered resource units
        collected_unit = self.collected_r - self.collected_r_before

        # calculating private info by weighting these
        self.I_priv = self.F_N * np.max(self.novelty) + self.F_R * collected_unit

    def evaluate_decision_processes(self):
        """updating inner decision processes according to the current state and the visual projection field"""
        w_p = self.w if self.w > self.T_w else 0
        u_p = self.u if self.u > self.T_u else 0
        dw = self.Eps_w * (np.mean(self.soc_v_field)) - self.g_w * (
                self.w - self.B_w) - u_p * self.S_uw  # self.tr_u() * self.S_uw
        du = self.Eps_u * self.I_priv - self.g_u * (self.u - self.B_u) - w_p * self.S_wu  # self.tr_w() * self.S_wu
        self.w += dw
        self.u += du
        if self.w > self.w_max:
            self.w = self.w_max
        if self.w < -self.w_max:
            self.w = -self.w_max
        if self.u > self.u_max:
            self.u = self.u_max
        if self.u < -self.u_max:
            self.u = -self.u_max

    def tr_w(self):
        """Relocation threshold function that checks if decision variable w is above T_w"""
        if self.w > self.T_w:
            return True
        else:
            return False

    def tr_u(self):
        """Exploitation threshold function that checks if decision variable u is above T_w"""
        if self.u > self.T_w:
            return True
        else:
            return False

### -------------------------- MODE FUNCTIONS -------------------------- ###
    
    def get_mode(self):
        """returning the current mode of the agent according to it's inner decision mechanisms as a human-readable
        string for external processes defined in the main simulation thread (such as collision that depends on the
        state of the at and also overrides it as it counts as ana emergency)"""
        if self.overriding_mode is None:
            if self.tr_w():
                return "relocate"
            else:
                return "explore"
        else:
            return self.overriding_mode

    def set_mode(self, mode):
        """setting the behavioral mode of the agent according to some human_readable flag. This can be:
            -explore
            -exploit
            -relocate
            -pool
            -collide"""
        if mode == "explore":
            # self.w = 0
            self.overriding_mode = None
        elif mode == "relocate":
            # self.w = self.T_w + 0.001
            self.overriding_mode = None
        elif mode == "collide":
            self.overriding_mode = "collide"
            # self.w = 0
        elif mode == "exploit":
            self.overriding_mode = "exploit"
            # self.w = 0
        elif mode == "pool":
            self.overriding_mode = "pool"
            # self.w = 0
        self.mode = mode

### -------------------------- PHYSICAL FUNCTIONS -------------------------- ###

    def prove_orientation(self):
        """Restricting orientation angle between 0 and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    def prove_velocity(self, velocity_limit=1):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if self.get_mode() == 'explore':
            if np.abs(self.velocity) > velocity_limit:
                # stopping agent if too fast during exploration
                # self.velocity = self.max_exp_vel # 1 ## not in humanexp8
                self.velocity = 1
                
    def reflect_from_walls(self):
        """
        implementing reflection conditions on environmental boundaries according to agent position
        """

        # Boundary conditions according to center of agent (simple)
        x = self.position[0] + self.radius
        y = self.position[1] + self.radius

        # Reflection from left wall
        if x < self.boundaries_x[0]:
            self.position[0] = self.boundaries_x[0] - self.radius

            if np.pi / 2 <= self.orientation < np.pi:
                self.orientation -= np.pi / 2
            elif np.pi <= self.orientation <= 3 * np.pi / 2:
                self.orientation += np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

        # Reflection from right wall
        if x > self.boundaries_x[1]:

            self.position[0] = self.boundaries_x[1] - self.radius - 1

            if 3 * np.pi / 2 <= self.orientation < 2 * np.pi:
                self.orientation -= np.pi / 2
            elif 0 <= self.orientation <= np.pi / 2:
                self.orientation += np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

        # Reflection from upper wall
        if y < self.boundaries_y[0]:
            self.position[1] = self.boundaries_y[0] - self.radius

            if np.pi / 2 <= self.orientation <= np.pi:
                self.orientation += np.pi / 2
            elif 0 <= self.orientation < np.pi / 2:
                self.orientation -= np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

        # Reflection from lower wall
        if y > self.boundaries_y[1]:
            self.position[1] = self.boundaries_y[1] - self.radius - 1
            if 3 * np.pi / 2 <= self.orientation <= 2 * np.pi:
                self.orientation += np.pi / 2
            elif np.pi <= self.orientation < 3 * np.pi / 2:
                self.orientation -= np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

##############################################################################
### -------------------------- MAIN UPDATE LOOP -------------------------- ###
##############################################################################
            
    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in the environment. These are not necessarily
                socially relevant, i.e. all agents.
        """
        # calculate socially relevant projection field (Vsoc and Vsoc+)
        self.calc_social_V_proj(agents)

        # calculate private information
        self.calc_I_priv()

        # update inner decision process according to visual field and private info
        self.evaluate_decision_processes()

        # CALCULATING velocity and orientation change according to inner decision process (dv)
        # we use if and not a + operator as this is less computationally heavy but the 2 is equivalent
        # vel, theta = int(self.tr_w()) * VSWRM_flocking_state_variables(...) + (1 - int(self.tr_w())) * random_walk(...)
        # or later when we define the individual and social forces
        # vel, theta = int(self.tr_w()) * self.F_soc(...) + (1 - int(self.tr_w())) * self.F_exp(...)
        if not self.get_mode() == "collide":
            if not self.tr_w() and not self.tr_u():
                # vel, theta = supcalc.random_walk(desired_vel=self.max_exp_vel) ## not in humanexp8
                vel, theta = supcalc.random_walk()
                self.set_mode("explore")
            elif self.tr_w() and self.tr_u():
                # if self.env_status == 1: ## not in humanexp8
                #     self.set_mode("exploit")
                #     vel, theta = (-self.velocity * self.exp_stop_ratio, 0)
                # else:
                #     vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field, v_desired=self.max_exp_vel)
                #     self.set_mode("relocate")
                self.set_mode("exploit")
                vel, theta = (-self.velocity * self.exp_stop_ratio, 0)
            elif self.tr_w() and not self.tr_u():
                # vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field, v_desired=self.max_exp_vel) ## not in humanexp8
                vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field)
                self.set_mode("relocate")
            elif self.tr_u() and not self.tr_w():
                # if self.env_status == 1: ## not in humanexp8
                #     self.set_mode("exploit")
                #     vel, theta = (-self.velocity * self.exp_stop_ratio, 0)
                # else:
                #     vel, theta = supcalc.random_walk(desired_vel=self.max_exp_vel)
                #     self.set_mode("explore")
                self.set_mode("exploit")
                vel, theta = (-self.velocity * self.exp_stop_ratio, 0)
        else:
            # COLLISION AVOIDANCE IS ACTIVE, let that guide us
            # As we don't have proximity sensor interface as with e.g. real robots we will let
            # the environment to enforce us into a collision maneuver from the simulation environment
            # so we don't change the current velocity from here.
            vel, theta = (0, 0)

        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # updating agent's state variables according to calculated vel and theta
            self.orientation += theta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += vel
            self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.position[0] += self.velocity * np.cos(self.orientation)
            self.position[1] -= self.velocity * np.sin(self.orientation)

            # boundary conditions if applicable
            self.reflect_from_walls()

        # updating agent visualization
        self.draw_update()
        self.collected_r_before = self.collected_r

### -------------------------- VISUALIZATION / HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        if self.get_mode() == "explore":
            self.color = colors.BLUE
        elif self.get_mode() == "flock" or self.get_mode() == "relocate":
            self.color = colors.PURPLE
        elif self.get_mode() == "collide":
            self.color = colors.RED
        elif self.get_mode() == "exploit":
            self.color = colors.GREEN
        elif self.get_mode() == "pool":
            self.color = colors.YELLOW

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # change agent color according to mode
        self.change_color()

        # update surface according to new orientation
        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        if self.is_moved_with_cursor:
            pygame.draw.circle(
                self.image, self.selected_color, (self.radius, self.radius), self.radius
            )
        else:
            pygame.draw.circle(
                self.image, self.color, (self.radius, self.radius), self.radius
            )

        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, colors.BACKGROUND, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius),
                         3)
        self.mask = pygame.mask.from_surface(self.image)

    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            if left_state:
                self.orientation += 0.1
            if right_state:
                self.orientation -= 0.1
            self.prove_orientation()
            self.is_moved_with_cursor = 1
            # updating agent visualization to make it more responsive
            self.draw_update()
        else:
            self.is_moved_with_cursor = 0