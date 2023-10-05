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
    def __init__(self, id, position, orientation, max_vel, 
                 FOV, vision_range, visual_exclusion, consumption, 
                 arch, model, RNN_type, NN_activ, boundary_info, radius, color):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param position: position of the agent in env as (x, y)
        :param orientation: absolute orientation of the agent (0: right, pi/2: up, pi: left, 3*pi/2: down)
        :param max_vel: 
        :param vis_field_res: resolution of the visual projection field of the agent in pixels
        :param FOV: visual field as a tuple of min max visible angles e.g. (-np.pi, np.pi)
        :param vision_range: in px the range/radius in which the agent is able to see other agents
        :param visual_exclusion: if True social cues can be visually excluded by non social cues
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
        self.collided_points = []

        # Neural network initialization / parameters
        (   CNN_input_size, 
            CNN_depths, 
            CNN_dims,               # last is num vis features fed to RNN
            RNN_other_input_size, 
            RNN_hidden_size, 
            LCL_output_size,        # dvel + dthetay
        ) = arch

        self.num_class_elements, vis_field_res = CNN_input_size

        # use given NNs to control agent or initialize new NNs
        if model: 
            self.model = model
        else: 
            from abm.NN.model import WorldModel
            # from abm.NN.model_simp import WorldModel
            self.model = WorldModel(arch=arch, activ=NN_activ, RNN_type=RNN_type)

            # print(f'Model Architecture: {arch}')
            param_vec_size = sum(p.numel() for p in self.model.parameters())
            # print(f'Total #Params: {param_vec_size}')
        
        # Visual field parameters
        self.vis_field_res = vis_field_res
        self.FOV = FOV
        # constructs array of each visually perceivable angle (- : left / + : right)
        self.phis = np.linspace(-FOV*np.pi, FOV*np.pi, vis_field_res)
        self.vision_range = vision_range
        self.visual_exclusion = visual_exclusion
        self.vis_field = [0] * vis_field_res

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
        # self.selected_color = colors.BLACK
        # self.show_stats = False
        # self.is_moved_with_cursor = 0

        # Initializing body + position
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        self.rect = self.image.get_rect(center = self.position + self.window_pad)

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
            angle_bw = supcalc.angle_bw_vis(self.vec_self_dir, vec_between, self.radius, distance)
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

                else: # shouldn't happen - used to be crash scenario - solved by clipped dot product
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
    #                 angle_bw = supcalc.angle_bw_vis(self.vec_self_dir, vec_between, self.radius, agent_distance)
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

        # Fill in vis_field with identification info for each visual perception ray
        self.fill_vis_field(self.vis_field_wall_dict, dict_type='walls')

### -------------------------- MOVEMENT FUNCTIONS -------------------------- ###

    def bind_orientation(self):
        """
        Restricts agent's orientation angle to [0 : 2*pi]
        """
        while self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        while self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

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

        # Update velocity (constrained by turn angle)
        self.velocity = self.max_vel * (1 - abs(NN_output))

        # Check for velocity-stopping collisions for each point of contact
        if self.mode == 'collide':
            for pt in self.collided_points:

                # calc vector between collided point + agent center
                vec_coll = pt - self.position - self.window_pad

                # calc orientation angle
                distance = np.linalg.norm(vec_coll)
                angle_coll = supcalc.angle_bw_coll(vec_coll, np.array([10,0]), distance, self.radius)
                # print(np.round(angle_coll,3), np.round(self.orientation,3))

                # check if collision angle is within 180d of current orientation + wrapping constraints
                if angle_coll + np.pi/2 > 2*np.pi:
                    if (angle_coll-np.pi/2) < self.orientation or (angle_coll+np.pi/2-2*np.pi) > self.orientation:
                        self.velocity = 0
                        # print('block wrap +')
                elif angle_coll - np.pi/2 < 0:
                    if (angle_coll+np.pi/2) > self.orientation or (angle_coll-np.pi/2+2*np.pi) < self.orientation:
                        self.velocity = 0
                        # print('block wrap -')
                else:
                    if (angle_coll-np.pi/2) < self.orientation < (angle_coll+np.pi/2):
                        self.velocity = 0
                        # print('block')

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
        pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius)
        pygame.draw.line(self.image, colors.BACKGROUND, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius), 3)
        self.rect = self.image.get_rect(center = self.position + self.window_pad)


    # def move_with_mouse(self, mouse, left_state, right_state):
    #     """Moving the agent with the mouse cursor, and rotating"""
    #     if self.rect.collidepoint(mouse):
    #         # setting position of agent to cursor position
    #         self.position = mouse - self.radius
    #         if left_state:
    #             self.orientation += 0.1
    #         if right_state:
    #             self.orientation -= 0.1
    #         self.prove_orientation()
    #         self.is_moved_with_cursor = 1
    #         # updating agent visualization to make it more responsive
    #         self.draw_update()
    #     else:
    #         self.is_moved_with_cursor = 0