"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""

from abm import colors
from abm.sprites import supcalc
# from abm.helpers import timer

import pygame
import numpy as np

class Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """
    # @timer
    def __init__(self, id, position, orientation, max_vel, collision_slowdown, 
                 FOV, vision_range, visual_exclusion, consumption, 
                 arch, model, RNN_type, NN_activ, boundary_info, radius, color):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param position: position of the agent in env as (x, y)
        :param orientation: absolute orientation of the agent (0: right, pi/2: up, pi: left, 3*pi/2: down)
        :param max_vel: 
        :param collision_slowdown: 
        :param vis_field_res: resolution of the visual projection field of the agent in pixels
        :param FOV: visual field as a tuple of min max visible angles e.g. (-np.pi, np.pi)
        :param vision_range: in px the range/radius in which the agent is able to see other agents
        :param visual_exclusion: if True social cues can be visually excluded by non social cues
        :param contact_field_res: resolution of the contact projection field of the agent in pixels
        :param consumption: (resource unit/time unit) consumption efficiency of agent
        :param NN: 
        :param boundary_info: 
        :param radius: radius of the agent in pixels
        :param color: color of the agent as (R, G, B)
        """
        # PyGame Sprite superclass
        super().__init__()

        # Unique parameters
        self.id = id 

        # Movement/behavior parameters
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.velocity = 0  # (absolute)
        
        self.pt_eye = np.array([
            self.position[0] + np.cos(orientation) * radius, 
            self.position[1] - np.sin(orientation) * radius])

        self.mode = "explore"  # explore / exploit / collide
        self.max_vel = max_vel
        self.collision_slowdown = collision_slowdown

        # Neural network initialization / parameters
        (   CNN_input_size, 
            CNN_depths, 
            CNN_dims,               # last is num vis features fed to RNN
            RNN_nonvis_input_size, 
            RNN_hidden_size, 
            LCL_output_size,        # dvel + dthetay
        ) = arch

        self.num_class_elements, vis_field_res = CNN_input_size
        contact_field_res, self.other_size = RNN_nonvis_input_size

        # use given NNs to control agent or initialize new NNs
        if model: 
            self.model = model
        else: 
            from abm.NN.model import WorldModel
            # from abm.NN.model_simp import WorldModel
            self.model = WorldModel(arch=arch, activ=NN_activ, RNN_type=RNN_type)

            print(f'Model Architecture: {arch}')
            param_vec_size = sum(p.numel() for p in self.model.parameters())
            print(f'Total #Params: {param_vec_size}')
        
        # Visual field parameters
        self.vis_field_res = vis_field_res
        self.FOV = FOV
        # constructs array of each visually perceivable angle (- : left / + : right)
        self.phis = np.linspace(-FOV*np.pi, FOV*np.pi, vis_field_res)
        self.vision_range = vision_range
        self.visual_exclusion = visual_exclusion
        self.vis_field = [0] * vis_field_res

        # Contact field parameters
        self.contact_field_res = contact_field_res
        # convention reflects orientation (0 : right, +pi/2 : up) / ending angle set to ensure even spacing around perimeter
        self.contact_phis = np.linspace(0, 2*np.pi - 2*np.pi/contact_field_res, contact_field_res)
        self.contact_field = [0] * contact_field_res

        # Resource parameters
        self.collected_r = 0  # resource units collected by agent 
        self.on_resrc = 0 # binary : whether agent is currently on top of a resource patch or not
        self.consumption = consumption

        # init placeholders for hidden activity + action for each sim timestep
        self.hidden = None
        self.action = 0

        # Environment related parameters
        self.x_min, self.x_max, self.y_min, self.y_max = boundary_info
        self.window_pad = 30
        # define names for each endpoint (top/bottom + left/right)
        self.boundary_endpts = [
            ('TL', np.array([ self.x_min, self.y_min ])),
            ('TR', np.array([ self.x_max, self.y_min ])),
            ('BL', np.array([ self.x_min, self.y_max ])),
            ('BR', np.array([ self.x_max, self.y_max ]))
        ]

        # Visualization / human interaction parameters
        self.radius = radius
        self.color = color
        self.selected_color = colors.BLACK

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
        self.rect.x, self.rect.y = self.position + self.window_pad
        self.mask = pygame.mask.from_surface(self.image)

### -------------------------- VISUAL FUNCTIONS -------------------------- ###

    def gather_self_percep_info(self):
        """
        update position/direction points + vector of self
        """
        # front-facing point on agent's perimeter according to its orientation
        self.pt_eye = np.array([
            self.position[0] + np.cos(self.orientation) * self.radius, 
            self.position[1] - np.sin(self.orientation) * self.radius])

        # direction vector, magnitude = radius, flipped y-axis
        self.vec_self_dir = self.pt_eye - self.position
        ## where v1[0] --> + : right, 0 : center, - : left, 10 : max
        ## where v1[1] --> + : down, 0 : center, - : up, 10 : max

    def gather_boundary_endpt_info(self):
        """
        create dictionary storing visually relevant information for each boundary endpoint
        """

        # print()
        # print(f'self \t {np.round(self.vec_self_dir/self.radius,2)} \t {np.round(self.orientation*90/np.pi,0)}')

        # print(np.round(self.phis*90/np.pi,0))

        self.boundary_endpt_dict = {}
        for endpt_name, endpt_coord in self.boundary_endpts:

            # calc vector between boundary endpoint + direction point (front-facing)
            vec_between = endpt_coord - self.pt_eye

            # calc magnitude/norm
            distance = np.linalg.norm(vec_between)

            # calc orientation angle
            angle_bw = supcalc.angle_between(self.vec_self_dir, vec_between, self.radius, distance)
            ## relative to perceiving agent, in radians between [+pi (left/CCW), -pi (right/CW)]

            # print(f'{endpt_name} \t {np.round(vec_between/distance,2)} \t {np.round(angle_bw*90/np.pi,0)}')

            # project to FOV
            proj = supcalc.find_nearest(self.phis, angle_bw)

            # update dictionary with added info
            self.boundary_endpt_dict[endpt_name] = (endpt_coord, distance, angle_bw, proj)

    # @timer
    def gather_boundary_wall_info(self):

        # initialize wall dict
        self.vis_field_wall_dict = {}
        # strings to name walls + call corresponding L/R endpts
        walls = [
            ('wall_north', 'TL', 'TR'),
            ('wall_south', 'BR', 'BL'),
            ('wall_east', 'TR', 'BR'),
            ('wall_west', 'BL', 'TL'),
        ]
        # unpack L/R angle limits of visual projection field
        phi_L_limit = self.phis[0]
        phi_R_limit = self.phis[-1]

        # loop over the 4 walls
        for wall_name, pt_L, pt_R in walls:

            # unpack dict entry for each corresponding endpt
            coord_L, _, angle_L, proj_L = self.boundary_endpt_dict[pt_L]
            coord_R, _, angle_R, proj_R = self.boundary_endpt_dict[pt_R]

            # if at least one endpt is visually perceivable, build dict entry
            if (phi_L_limit <= angle_L <= phi_R_limit) or (phi_L_limit <= angle_R <= phi_R_limit): 

                # print(f'see \t {wall_name} \t {np.round(angle_L*90/np.pi,0), np.round(angle_R*90/np.pi,0)}')

                self.vis_field_wall_dict[wall_name] = {}
                self.vis_field_wall_dict[wall_name]['coord_L'] = coord_L
                self.vis_field_wall_dict[wall_name]['coord_R'] = coord_R

                self.vis_field_wall_dict[wall_name]['angle_L'] = angle_L
                self.vis_field_wall_dict[wall_name]['angle_R'] = angle_R

                if proj_L <= proj_R:

                    self.vis_field_wall_dict[wall_name]['proj_L'] = proj_L
                    self.vis_field_wall_dict[wall_name]['proj_R'] = proj_R

                # L/R edge cases (angle sign switches behind the agent)
                else: # proj_L > proj_R 

                    # L-endpoint is far enough L it becomes R (positive + end of visual field)
                    if proj_L == len(self.phis)-1:
                        self.vis_field_wall_dict[wall_name]['proj_L'] = 0
                        self.vis_field_wall_dict[wall_name]['proj_R'] = proj_R

                    else: # R-endpoint is far enough R it becomes L (negative + start of visual field)
                        self.vis_field_wall_dict[wall_name]['proj_L'] = proj_L
                        self.vis_field_wall_dict[wall_name]['proj_R'] = len(self.phis)-1


        if not self.vis_field_wall_dict: # no walls had endpoints within FOV - find closest wall via L endpoints

            # list endpoints left of L perception limit
            nonvis_L_endpts = [(name,angle) for name,(_,_,angle,_) in self.boundary_endpt_dict.items() if angle < phi_L_limit]

            if nonvis_L_endpts:
                # return endpt name with max angle of these L angles
                closest_nonvis_L_endpt_name,_ = max(nonvis_L_endpts, key = lambda t: t[1])
                # find corresponding wall_name + R endpt
                for wall_name, pt_L, pt_R in walls:
                    if pt_L == closest_nonvis_L_endpt_name:
                        break
                # print(f'see \t {wall_name} \t {np.round(angle_L*90/np.pi,0), np.round(angle_R*90/np.pi,0)}')
                # create dict entry for this wall
                coord_L, _, angle_L, proj_L = self.boundary_endpt_dict[pt_L]
                coord_R, _, angle_R, proj_R = self.boundary_endpt_dict[pt_R]
                self.vis_field_wall_dict[wall_name] = {}
                self.vis_field_wall_dict[wall_name]['coord_L'] = coord_L
                self.vis_field_wall_dict[wall_name]['coord_R'] = coord_R
                self.vis_field_wall_dict[wall_name]['angle_L'] = angle_L
                self.vis_field_wall_dict[wall_name]['angle_R'] = angle_R
                self.vis_field_wall_dict[wall_name]['proj_L'] = proj_L
                self.vis_field_wall_dict[wall_name]['proj_R'] = proj_R
            
            else:
                # list endpoints left of R perception limit
                nonvis_R_endpts = [(name,angle) for name,(_,_,angle,_) in self.boundary_endpt_dict.items() if angle > phi_R_limit]
                
                if nonvis_R_endpts:
                    # return endpt name with max angle of these R angles
                    closest_nonvis_R_endpt_name,_ = min(nonvis_R_endpts, key = lambda t: t[1])
                    # find corresponding wall_name + L endpt
                    for wall_name, pt_L, pt_R in walls:
                        if pt_R == closest_nonvis_R_endpt_name:
                            break
                    # print(f'see \t {wall_name} \t {np.round(angle_L*90/np.pi,0), np.round(angle_R*90/np.pi,0)}')
                    # create dict entry for this wall
                    coord_L, _, angle_L, proj_L = self.boundary_endpt_dict[pt_L]
                    coord_R, _, angle_R, proj_R = self.boundary_endpt_dict[pt_R]
                    self.vis_field_wall_dict[wall_name] = {}
                    self.vis_field_wall_dict[wall_name]['coord_L'] = coord_L
                    self.vis_field_wall_dict[wall_name]['coord_R'] = coord_R
                    self.vis_field_wall_dict[wall_name]['angle_L'] = angle_L
                    self.vis_field_wall_dict[wall_name]['angle_R'] = angle_R
                    self.vis_field_wall_dict[wall_name]['proj_L'] = proj_L
                    self.vis_field_wall_dict[wall_name]['proj_R'] = proj_R

                else:
                    # parameters = nans from exploding gradient
                    # gradient clipping not a viable method for RL
                    # --> send signal to parent function + crash simulation + return 0 fitness
                    self.vis_field_wall_dict = None

    # def gather_agent_info(self, agents):

    #     # initialize agent dict
    #     self.vis_field_agent_dict = {}

    #     # for all agents in the simulation
    #     for ag in agents:

    #         # exclude self from list
    #         agent_id = ag.id
    #         if agent_id != self.id:

    #             # exclude agents outside range of vision (calculate distance bw agent center + self eye)
    #             agent_coord = ag.position + self.radius
    #             vec_between = agent_coord - self.pt_eye
    #             agent_distance = np.linalg.norm(vec_between)
    #             if agent_distance <= self.vision_range:
                    
    #                 # exclude agents outside FOV limits (calculate visual boundaries of agent)

    #                 # orientation angle relative to perceiving agent, in radians between [+pi (left/CCW), -pi (right/CW)]
    #                 angle_bw = supcalc.angle_between(self.vec_self_dir, vec_between, self.radius, agent_distance)
    #                 # exclusionary angle between agent + self, taken to L/R boundaries
    #                 angle_edge = np.arctan(self.radius / agent_distance)
    #                 angle_L = angle_bw - angle_edge
    #                 angle_R = angle_bw + angle_edge
    #                 # unpack L/R angle limits of visual projection field
    #                 phi_L_limit = self.phis[0]
    #                 phi_R_limit = self.phis[-1]
    #                 if (phi_L_limit <= angle_L <= phi_R_limit) or (phi_L_limit <= angle_R <= phi_R_limit): 

    #                     # find projection endpoints
    #                     proj_L = supcalc.find_nearest(self.phis, angle_L)
    #                     proj_R = supcalc.find_nearest(self.phis, angle_R)

    #                     # calculate left edgepoint on agent's perimeter according to the angle in which it is perceived
    #                     coord_L = np.array([
    #                         agent_coord[0] + np.cos(self.orientation - angle_bw + np.pi/2) * self.radius, 
    #                         agent_coord[1] - np.sin(self.orientation - angle_bw + np.pi/2) * self.radius])
    #                     # exploiting symmetry to find right edge
    #                     vec_agent_L_edge = agent_coord - coord_L
    #                     coord_R = agent_coord + vec_agent_L_edge
                    
    #                     # update dictionary with all relevant info
    #                     self.vis_field_agent_dict['agent_'+id] = {}
    #                     self.vis_field_agent_dict['agent_'+id]['coord_center'] = agent_coord
    #                     self.vis_field_agent_dict['agent_'+id]['mode'] = ag.get_mode()
    #                     self.vis_field_agent_dict['agent_'+id]['distance'] = agent_distance
    #                     self.vis_field_agent_dict['agent_'+id]['angle'] = angle_bw
    #                     self.vis_field_agent_dict['agent_'+id]['proj_L'] = proj_L
    #                     self.vis_field_agent_dict['agent_'+id]['proj_R'] = proj_R
    #                     self.vis_field_agent_dict['agent_'+id]['proj_L_ex'] = proj_L
    #                     self.vis_field_agent_dict['agent_'+id]['proj_R_ex'] = proj_R
    #                     self.vis_field_agent_dict['agent_'+id]['coord_L'] = coord_L
    #                     self.vis_field_agent_dict['agent_'+id]['coord_R'] = coord_R

    #                     # key differences -->
    #                     # dict includes both agent + boundary wall info
    #                     # calculates visual exclusions from all agents
    #                         # also includes those on the same exploited patch (patchwise_social_exclusion)
    #                     # includes non-exploiting agents in end perception

    # def calculate_perceptual_exclusions(self):
    #     """
    #     Iterates over pairs of obstacles, excluding occluded obstacles/parts
    #     """
    #     # filter visual dict for agents + rank according to distance from self (low first)
    #     agent_by_dist_info = sorted(self.vis_field_agent_dict, key=lambda kv: kv[1]['distance'])
    #     self.vis_field_agent_dict = OrderedDict(agent_by_dist_info)

    #     # combination operation, though itertools doesn't seem to provide performance benefit and is less understandable
    #     for id_close, obs_close in self.vis_field_agent_dict.items():
    #         for id_far, obs_far in self.vis_field_agent_dict.items():
    #             if obs_far["distance"] > obs_close["distance"]: 
    #                 # Partial R-side exclusion
    #                 if obs_far["proj_R_ex"] > obs_close["proj_L"] > obs_far["proj_L_ex"]: 
    #                     obs_far["proj_R_ex"] = obs_close["proj_L"]
    #                     continue
    #                 # Partial L-side exclusion
    #                 if obs_far["proj_L_ex"] < obs_close["proj_R"] < obs_far["proj_R_ex"]:
    #                     obs_far["proj_L_ex"] = obs_close["proj_R"]
    #                     continue
    #                 # Total exclusion
    #                 if obs_close["proj_L"] <= obs_far["proj_L_ex"] and obs_close["proj_R"] >= obs_far["proj_R_ex"]:
    #                     obs_far["proj_L_ex"] = -1
    #                     obs_far["proj_R_ex"] = -1

    def fill_vis_field(self, proj_dict, dict_type=None):
        """
        Mark projection field according to each wall or agent
        """
        # # pull relevant info from dict ---> loops over walls ---> can result in error at border points since it does not check
        # for obj_name, v in proj_dict.items():
        #     if dict_type == 'walls':
        #         phi_from = v["proj_L"]
        #         phi_to = v["proj_R"]
        #     else: # dict_type == 'agents
        #         phi_from = v["proj_L_ex"]
        #         phi_to = v["proj_R_ex"]
        #         obj_name = 'agent_' + v["mode"] # uses agent_expl or agent_wand for vis_field value instead of "agent_[id]"
        #     # raycast to wall/agent for each discretized perception angle within FOV range
        #     # fill in relevant identification information (wall name / agent mode)
        #     for i in range(phi_from, phi_to + 1):
        #         self.vis_field[i] = obj_name


        # raycast to wall/agent for each discretized perception angle within FOV range
        # fill in relevant identification information (wall name / agent mode)
        for i in range(self.vis_field_res):

            # look for intersections
            for obj_name, v in proj_dict.items():
                if v["angle_L"] <= self.phis[i] <= v["angle_R"]:
                    # if dict_type == 'agents': obj_name = 'agent_' + v["mode"] # uses agent_expl or agent_wand for vis_field value instead of "agent_[id]"
                    self.vis_field[i] = obj_name
            
            # no intersections bc one endpoint is behind back, iterate again
            if self.vis_field[i] == 0:
                for obj_name, v in proj_dict.items():
                    # if dict_type == 'agents': obj_name = 'agent_' + v["mode"] # uses agent_expl or agent_wand for vis_field value instead of "agent_[id]"
                    if v["angle_L"] > v["angle_R"]:
                        self.vis_field[i] = obj_name
        
        # print(self.vis_field)

    # @timer
    def visual_sensing(self):
        """
        Accumulates visual sensory functions
        """
        # Zero vis_field from previous step
        self.vis_field = [0] * self.vis_field_res
        # Gather relevant info for self / boundary endpoints / walls
        self.gather_self_percep_info()
        self.gather_boundary_endpt_info()
        self.gather_boundary_wall_info()

        # crash condition --> agent isn't able to create visual projection field dictionary
        if self.vis_field_wall_dict is None: return True

        # Fill in vis_field with identification info for each visual perception ray
        self.fill_vis_field(self.vis_field_wall_dict, dict_type='walls')

### -------------------------- MOVEMENT FUNCTIONS -------------------------- ###

    def block_angle(self, angle):
        """
        Updates contact_field (bool) + blocked_angle (float) with the agent's currently blocked direction
        """
        index = supcalc.find_nearest(self.contact_phis, angle)
        self.contact_field[index] = 1
        self.blocked_angles.append(angle)

    # @timer
    def wall_contact_sensing(self):
        """
        Signals blocked directions to the agent for all boundary walls
        """
        # Call agent center coordinates
        x, y = self.position

        # Zero contact_field + block_angles lists from previous step
        self.contact_field = [0] * self.contact_field_res
        self.blocked_angles = []

        if y < self.y_min + self.radius*2: # top wall
            self.block_angle(np.pi/2)
        if y > self.y_max - self.radius*2: # bottom wall
            self.block_angle(3*np.pi/2)
        if x < self.x_min + self.radius*2: # left wall
            self.block_angle(np.pi)
        if x > self.x_max - self.radius*2: # right wall
            self.block_angle(0)

    def bind_velocity(self):
        """
        Restricts agent's absolute velocity to [0 : max_vel]
        """
        if self.velocity < 0:
            self.velocity = 0
        if self.velocity > self.max_vel:
            self.velocity = self.max_vel

    def bind_orientation(self):
        """
        Restricts agent's orientation angle to [0 : 2*pi]
        """
        while self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        while self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    # def wall_collision_reflection(self):
    #     """
    #     Implements simple reflection conditions on environmental boundaries according to orientation
    #     """
    #     # Agents set as non-colliding until proven otherwise
    #     self.mode = "explore"
    #     collision = False

    #     for angle in self.blocked_angles:

    #         if angle == np.pi/2 and 0 < self.orientation < np.pi: # top wall
    #             self.orientation = -self.orientation
    #             self.bind_orientation()
    #             collision = True

    #         if angle == 3*np.pi/2 and np.pi < self.orientation < 2*np.pi: # bottom wall
    #             self.orientation = -self.orientation
    #             self.bind_orientation()
    #             collision = True

    #         if angle == np.pi and np.pi/2 < self.orientation < 3*np.pi/2: # left wall
    #             self.orientation = np.pi - self.orientation
    #             self.bind_orientation()
    #             collision = True

    #         if angle == 0 and (3*np.pi/2 < self.orientation or self.orientation < np.pi/2): # right wall
    #             self.orientation = np.pi - self.orientation
    #             self.bind_orientation()
    #             collision = True

    #     if collision is True:
    #         # Collision absorbs the speed by specified ratio + changes agent mode
    #         self.velocity = self.velocity * self.collision_slowdown
    #         self.mode = "collide"

    # def wall_collision_absorption(self):
    #     """
    #     Implements simple absorption conditions on environmental boundaries according to orientation
    #     - x/y component of velocity is reduced to zero upon impact with a vertical/horizontal wall, respectively
    #     - orientation is unchanged
    #     - moving_dir used to calculate position at next time step without changing agent orientation
    #     - mode is set to 'collide'
    #     """
    #     # Set agent as non-colliding (exploring at max velocity) until proven otherwise
    #     self.velocity = self.max_vel
    #     moving_dir = None
    #     self.mode = "explore"

    #     for angle in self.blocked_angles:

    #         if angle == np.pi/2 and 0 < self.orientation < np.pi: # top wall
    #             self.velocity -= np.abs( np.sin(self.orientation) ) * self.velocity # v_y = 0
    #             if self.orientation < np.pi/2:
    #                 moving_dir = 0
    #             else:
    #                 moving_dir = np.pi
    #             self.mode = "collide"

    #         if angle == 3*np.pi/2 and np.pi < self.orientation < 2*np.pi: # bottom wall
    #             self.velocity -= np.abs( np.sin(self.orientation) ) * self.velocity # v_y = 0
    #             if self.orientation > 3*np.pi/2:
    #                 moving_dir = 0
    #             else:
    #                 moving_dir = np.pi
    #             self.mode = "collide"

    #         if angle == np.pi and np.pi/2 < self.orientation < 3*np.pi/2: # left wall

    #             if moving_dir: # agent is next to top or bottom wall
    #                 self.velocity = 0
    #             else:
    #                 self.velocity -= np.abs( np.cos(self.orientation) ) * self.velocity # v_x = 0
    #                 if self.orientation < np.pi:
    #                     moving_dir = np.pi/2
    #                 else:
    #                     moving_dir = 3*np.pi/2
    #                 self.mode = "collide"

    #         if angle == 0 and (3*np.pi/2 < self.orientation or self.orientation < np.pi/2): # right wall
                
    #             if moving_dir: # agent is next to top or bottom wall
    #                 self.velocity = 0
    #             else:
    #                 self.velocity -= np.abs( np.cos(self.orientation) ) * self.velocity # v_x = 0
    #                 if self.orientation < np.pi/2:
    #                     moving_dir = np.pi/2
    #                 else:
    #                     moving_dir = np.pi
    #                 self.mode = "collide"
        
    #     print(self.velocity, moving_dir)
    #     return moving_dir

    def wall_collision_sticky(self):
        """
        Implements sticky conditions on environmental boundaries according to orientation
        - v = 0 when orientation is facing into a wall
        - agent only allowed to move when orientation changes
        - mode is set to 'collide'
        """
        # Set agent as non-colliding (exploring at max velocity) until proven otherwise
        self.velocity = self.max_vel
        self.mode = "explore"

        for angle in self.blocked_angles:

            if angle == np.pi/2 and 0 < self.orientation < np.pi: # top wall
                self.velocity = 0
                self.mode = "collide"

            if angle == 3*np.pi/2 and np.pi < self.orientation < 2*np.pi: # bottom wall
                self.velocity = 0
                self.mode = "collide"

            if angle == np.pi and np.pi/2 < self.orientation < 3*np.pi/2: # left wall
                self.velocity = 0
                self.mode = "collide"

            if angle == 0 and (3*np.pi/2 < self.orientation or self.orientation < np.pi/2): # right wall
                self.velocity = 0
                self.mode = "collide"

    # @timer
    def move(self, NN_output):
        """
        Incorporates NN outputs (change in orientation, or velocity + orientation)
        Calculates next position with collisions as absorbing boundary conditions
        """
        # NN output via tanh scales to a range of [-1 : 1]
        # Scale to max 90 deg turns [-pi/2 : pi/2] per timestep
        turn = NN_output * np.pi / 2

        # Shift orientation accordingly + bind to [0 : 2pi]
        self.orientation += turn
        # self.orientation += np.pi/16
        self.bind_orientation()

        # Update velocity
        # self.velocity += dvel
        # self.bind_velocity()  # to [0 : max_vel]
        self.velocity = self.max_vel * (1 - abs(NN_output))

        # Impelement collision boundary conditions
        self.wall_collision_sticky()

        # Calculate agent's next position
        orient_comp = np.array((
            np.cos(self.orientation),
            -np.sin(self.orientation)
        ))
        self.position += self.velocity * orient_comp

### -------------------------- NEURAL NETWORK FUNCTIONS -------------------------- ###

    def encode_one_hot(self, field):
        """
        one hot encode the visual field according to class indices:
            single-agent: (wall_east, wall_north, wall_west, wall_south)
            multi-agent: (wall_east, wall_north, wall_west, wall_south, agent_expl, agent_nonexpl)
        """
        field_onehot = np.zeros((self.num_class_elements, len(field)))

        for i,x in enumerate(field):
            if x == 'wall_north': field_onehot[0,i] = 1
            elif x == 'wall_south': field_onehot[1,i] = 1
            elif x == 'wall_east': field_onehot[2,i] = 1
            else: # x == 'wall_west'
                field_onehot[3,i] = 1
            # elif x == 'agent_expl': field_onehot[i,4] = 1
            # else: # x == 'agent_nonexpl
            #     field_onehot[i,5] = 1
        return field_onehot

    # @timer
    def assemble_NN_inputs(self):

        # transform visual data to onehot encoding --> matrix passed directly to CNN
        vis_input = self.encode_one_hot(self.vis_field)

        # store contact with food/proprio data as 1D array
        other_input = np.zeros(self.contact_field_res + self.other_size)
        other_input[:self.contact_field_res] = self.contact_field
        # other_input[:self.contact_field_res] = np.zeros(self.contact_field_res) # no contact

        # *ordered in chance of occurence*
        if self.other_size == 3:   # last action + last movement + food presence
            other_input[self.contact_field_res:] = np.array([ self.action, self.velocity / self.max_vel, self.on_resrc ]) 
            # other_input[self.contact_field_res:] = np.array([ 0, 1, 0 ]) # no proprio
            # other_input[self.contact_field_res:] = np.array([ self.action, 1, self.on_resrc ]) # no speed
            # other_input[self.contact_field_res:] = np.array([ 0, self.velocity / self.max_vel, self.on_resrc ]) # 0turn
            # other_input[self.contact_field_res:] = np.array([ .5, self.velocity / self.max_vel, self.on_resrc ]) # halfturn
            # other_input[self.contact_field_res:] = np.array([ 1, self.velocity / self.max_vel, self.on_resrc ]) # 1turn
            # other_input[self.contact_field_res:] = np.array([ self.action, self.velocity / self.max_vel, 0 ]) # no food
        elif self.other_size == 0: # none
            pass
        elif self.other_size == 2: # last action + last movement
            other_input[self.contact_field_res:] = np.array([ self.action, self.velocity / self.max_vel ]) 
        elif self.other_size == 1: # last action
            other_input[self.contact_field_res:] = np.array([ self.action ]) 
        else: raise Exception('NN_input_other_size not valid')

        # mask = [0,0,0,0,1,1,1] # no contact
        # mask = [1,1,1,1,1,1,0] # no food
        # mask = [1,1,1,1,0,1,1] # no angle
        # mask = [1,1,1,1,1,0,1] # no speed
        # other_input = np.array([x*y for x,y in zip(other_input,mask)])

        return vis_input, other_input

### -------------------------- VISUALIZATION / HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        if self.mode == "explore":
            self.color = colors.BLUE
        elif self.mode == "exploit":
            self.color = colors.GREEN
        elif self.mode == "collide":
            self.color = colors.RED
    # @timer
    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
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
        self.rect.centerx, self.rect.centery = self.position + self.window_pad

        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, colors.BACKGROUND, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius),
                         3)
        self.mask = pygame.mask.from_surface(self.image)

    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position = mouse - self.radius
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