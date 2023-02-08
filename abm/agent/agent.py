"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""

import pygame
import numpy as np
from abm.contrib import colors, decision_params, movement_params
from abm.agent import supcalc
from abm.NN.CTRNN import CTRNN
import importlib

class Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self, id, position, orientation, max_vel, collision_slowdown, vis_field_res, FOV, vision_range, visual_exclusion, 
                 contact_field_res, consumption, NN, NN_weight_init, vis_size, contact_size, NN_input_size, NN_hidden_size, 
                 NN_output_size, boundary_info, radius, color, print_enabled):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param position: position of the agent in env as (x, y)
        :param max_vel: 
        :param collision_slowdown: 
        :param orientation: absolute orientation of the agent (0: right, pi/2: up, pi: left, 3*pi/2: down)
        :param vis_field_res: resolution of the visual projection field of the agent in pixels
        :param FOV: visual field as a tuple of min max visible angles e.g. (-np.pi, np.pi)
        :param vision_range: in px the range/radius in which the agent is able to see other agents
        :param visual_exclusion: if True social cues can be visually excluded by non social cues
        :param contact_field_res: resolution of the contact projection field of the agent in pixels
        :param consumption: (resource unit/time unit) consumption efficiency of agent
        :param NN: 
        :param NN_weight_init: 
        :param vis_size: 
        :param contact_size: 
        :param NN_input_size: 
        :param NN_hidden_size: 
        :param NN_output_size: 
        :param boundary_info: 
        :param radius: radius of the agent in pixels
        :param color: color of the agent as (R, G, B)
        """
        # PyGame Sprite superclass
        super().__init__()

        # in case we run multiple simulations, we reload the env parameters
        importlib.reload(decision_params)
        importlib.reload(movement_params)

        # Unique parameters
        self.id = id 

        # Movement/behavior parameters
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.velocity = 0  # (absolute)
        
        self.pt_center = self.position + radius
        self.pt_eye = np.array([
            self.pt_center[0] + np.cos(orientation) * radius, 
            self.pt_center[1] - np.sin(orientation) * radius])

        self.mode = "explore"  # explore / exploit / collide
        self.max_vel = max_vel
        self.collision_slowdown = collision_slowdown
        
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

        # Neural network initialization / parameters
        self.vis_size = vis_size
        self.contact_size = contact_size
        self.other_size = NN_input_size - vis_size - contact_size
        self.input_size = NN_input_size
        NN_arch = (NN_input_size, NN_hidden_size, NN_output_size)
        # use given NN to control agent or initialize a new NN
        if NN: self.NN = NN
        else:  self.NN = CTRNN(architecture=NN_arch, init=NN_weight_init, dt=100) 
        self.hidden = None # to store hidden activity each time step

        # Environment related parameters
        self.x_min, self.x_max, self.y_min, self.y_max = boundary_info
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
        self.rect.x, self.rect.y = self.position
        self.mask = pygame.mask.from_surface(self.image)

### -------------------------- VISUAL FUNCTIONS -------------------------- ###

    def gather_self_percep_info(self):
        """
        gather relevant perception info of self (viewing point / direction vector)
        """
        # center point of self
        self.pt_center = self.position + self.radius

        # front-facing point on agent's perimeter according to its orientation
        self.pt_eye = np.array([
            self.pt_center[0] + np.cos(self.orientation) * self.radius, 
            self.pt_center[1] - np.sin(self.orientation) * self.radius])

        # direction vector, magnitude = radius, flipped y-axis
        self.vec_self_dir = self.pt_eye - self.pt_center
        ## where v1[0] --> + : right, 0 : center, - : left, 10 : max
        ## where v1[1] --> + : down, 0 : center, - : up, 10 : max

    def gather_boundary_endpt_info(self):
        """
        create dictionary storing visually relevant information for each boundary endpoint
        """
        self.boundary_endpt_dict = {}
        for endpt_name, endpt_coord in self.boundary_endpts:

            # calc vector between boundary endpoint + self eye (front-facing point)
            vec_between = endpt_coord - self.pt_eye

            # calc magnitude/norm
            distance = np.linalg.norm(vec_between)

            # calc orientation angle
            angle_bw = supcalc.angle_between(self.vec_self_dir, vec_between, self.radius, distance)
            ## relative to perceiving agent, in radians between [+pi (left/CCW), -pi (right/CW)]

            # project to FOV
            proj = supcalc.find_nearest(self.phis, angle_bw)

            # update dictionary with added info
            self.boundary_endpt_dict[endpt_name] = (endpt_coord, distance, angle_bw, proj)

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
        # pull relevant info from dict
        for obj_name, v in proj_dict.items():

            if dict_type == 'walls':
                phi_from = v["proj_L"]
                phi_to = v["proj_R"]
            else: # dict_type == 'agents
                phi_from = v["proj_L_ex"]
                phi_to = v["proj_R_ex"]
                obj_name = 'agent_' + v["mode"] # uses agent_expl or agent_wand for vis_field value instead of "agent_[id]"

            # raycast to wall/agent for each discretized perception angle within FOV range
            # fill in relevant identification information (wall name / agent mode)
            for i in range(phi_from, phi_to + 1):
                
                self.vis_field[i] = obj_name

    def visual_sensing(self):
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

    def wall_contact_sensing(self):
        """
        Signals blocked directions to the agent for all boundary walls
        """
        # Call agent center coordinates
        x, y = self.pt_center

        # Zero contact_field + block_angles lists from previous step
        self.contact_field = [0] * self.contact_field_res
        self.blocked_angles = []

        if y < self.y_min: # top wall
            self.block_angle(np.pi/2)
        if y > self.y_max: # bottom wall
            self.block_angle(3*np.pi/2)
        if x < self.x_min: # left wall
            self.block_angle(np.pi)
        if x > self.x_max: # right wall
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

    def wall_collision_impact(self):
        """
        Implements simple reflection conditions on environmental boundaries according to orientation
        """
        # Agents set as non-colliding until proven otherwise
        self.mode = "explore"
        collision = False

        for angle in self.blocked_angles:

            if angle == np.pi/2 and 0 < self.orientation < np.pi: # top wall
                self.orientation = -self.orientation
                self.bind_orientation()
                collision = True

            if angle == 3*np.pi/2 and np.pi < self.orientation < 2*np.pi: # bottom wall
                self.orientation = -self.orientation
                self.bind_orientation()
                collision = True

            if angle == np.pi and np.pi/2 < self.orientation < 3*np.pi/2: # left wall
                self.orientation = np.pi - self.orientation
                self.bind_orientation()
                collision = True

            if angle == 0 and (3*np.pi/2 < self.orientation < 2*np.pi or 0 < self.orientation < np.pi/2): # right wall
                self.orientation = np.pi - self.orientation
                self.bind_orientation()
                collision = True

        if collision is True:
            # Collision absorbs the speed by specified ratio + changes agent mode
            self.velocity = self.velocity * self.collision_slowdown
            self.mode = "collide"

    def move(self, actions):
        """
        Incorporates NN outputs (change in velocity + orientation)
        Calculates next position with collisions as absorbing boundary conditions
        """
        # Unpack actions (NN outputs, tanh scales possible range to [-1 : +1])
        dvel, dtheta = actions

        # Update + bound velocity/orientation
        self.velocity += dvel
        self.bind_velocity()  # to [0 : max_vel]
        self.orientation += dtheta
        self.bind_orientation()  # to [0 : 2pi]

        # Impelement collision boundary conditions (reflect orientation + absorb velocity)
        self.wall_collision_impact()

        # Calculate agent's next position
        self.position[0] += self.velocity * np.cos(self.orientation)
        self.position[1] -= self.velocity * np.sin(self.orientation)

### -------------------------- NEURAL NETWORK FUNCTIONS -------------------------- ###

    def encode_one_hot(self, field):
        """
        one hot encode the visual field according to class indices:
            single-agent: (wall_north, wall_south, wall_east, wall_west)
            multi-agent: (wall_north, wall_south, wall_east, wall_west, agent_expl, agent_nonexpl)
        """
        field_onehot = np.zeros((len(field), 4))
        # field_onehot = np.zeros((len(field), 6))

        for i,x in enumerate(field):

            if x == 'wall_north': field_onehot[i,0] = 1
            elif x == 'wall_south': field_onehot[i,1] = 1
            elif x == 'wall_east': field_onehot[i,2] = 1
            else: # x == 'wall_west'
                field_onehot[i,3] = 1
            # elif x == 'agent_expl': field_onehot[i,4] = 1
            # else: # x == 'agent_nonexpl
            #     field_onehot[i,5] = 1

        return field_onehot

    def assemble_NN_inputs(self):

        # transform visual/contact data to onehot encoding
        vis_field_onehot = self.encode_one_hot(self.vis_field)
        contact_field_onehot = self.encode_one_hot(self.contact_field)

        # scale velocity + orientation to [0 : 1] boolean interval similar to visual/contact fields
        vel_scaled = self.velocity / self.max_vel
        orient_scaled = self.orientation / (2*np.pi)

        # store data as 1D array
        NN_input = np.zeros(self.input_size)
        NN_input[0:self.vis_size] = vis_field_onehot.transpose().flatten()
        NN_input[self.vis_size : self.vis_size + self.contact_size] = contact_field_onehot.transpose().flatten()

        # NN_input[-self.other_size :] = np.array([ self.on_resrc, self.velocity, self.orientation ])
        NN_input[-self.other_size :] = np.array([ vel_scaled, orient_scaled ])
        # NN_input[-self.other_size :] = np.array([ self.on_resrc, vel_scaled, orient_scaled ])

        return NN_input

### -------------------------- VISUALIZATION / HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        if self.mode == "explore":
            self.color = colors.BLUE
        elif self.mode == "exploit":
            self.color = colors.GREEN
        elif self.mode == "collide":
            self.color = colors.RED

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x, self.rect.y = self.position

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