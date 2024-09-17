import contextlib
with contextlib.redirect_stdout(None): # blocks pygame initialization messages
    import pygame

import numpy as np
import sys
import time

from abm import colors
from abm.sprites import supcalc
from abm.sprites.agent import Agent
from abm.sprites.resource import Resource
from abm.sprites.wall import Wall
from abm.monitoring import tracking, plot_funcs
# from abm.monitoring.screen_recorder import ScreenRecorder
# from abm.helpers import timer

class Simulation:
    # @timer
    def __init__(self, env_size, window_pad,
                 N, T, with_visualization, framerate, print_enabled, plot_trajectory, log_zarr_file, save_ext,
                 agent_radius, max_vel, vis_field_res, vision_range, agent_fov, show_vision_range, agent_consumption, 
                 N_res, patch_radius, res_pos, res_units, res_quality, regenerate_patches, 
                 NN, other_input, vis_transform, percep_angle_noise_std, percep_dist_noise_std, action_noise_std,
                #  landmark_radius, LM_dist_noise_std, LM_angle_noise_std, LM_radius_noise_std, 
                mode, views=None, x=None, y=None,
                 ):
        """
        Initializing the main simulation instance
        :param width: real width of environment (not window size)
        :param height: real height of environment (not window size)
        :param window_pad: padding of the environment in simulation window in pixels
        :param N: number of agents
        :param T: simulation time
        :param with_visualization: turns visualization on or off. For large batch autmatic simulation should be off so
            that we can use a higher/maximal framerate
        :param framerate: framerate of simulation
        :param print_enabled:
        :param plot_trajectory:
        :param log_zarr_file:
        :param save_ext:
        :param agent_radius: radius of the agents
        :param max_vel:
        :param vis_field_res: projection field (visual + proximity) resolution in pixels
        :param vision_range: range (in px) of agents' vision
        :param agent_fov (float): the field of view of the agent as percentage. e.g. if 0.5, the the field of view is
                                between -pi/2 and pi/2
        :param show_vision_range: bool to switch visualization of visual range for agents. If true the limit of far
                                and near field visual field will be drawn around the agents
        :param agent_consumption: agent consumption (exploitation speed) in res. units / time units
        :param N_res: number of resource patches in the environment
        :param patch_radius: radius of resource patches
        :param min_res_perpatch: minimum resource unit per patch
        :param max_res_perpatch: maximum resource units per patch
        :param min_res_quality: minimum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        : param max_res_quality: maximum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        :param regenerate_patches: bool to decide if patches shall be regenerated after depletion
        :param NN:
        """
        # Arena parameters
        self.WIDTH, self.HEIGHT = env_size
        self.window_pad = window_pad
        self.coll_boundary_thickness = agent_radius

        self.x_min, self.x_max = 0, self.WIDTH
        self.y_min, self.y_max = 0, self.HEIGHT
        
        self.boundary_info = (0, self.WIDTH, 
                              0, self.HEIGHT)
        self.boundary_info_coll = (agent_radius*2, self.WIDTH - agent_radius*2, 
                                   agent_radius*2, self.HEIGHT - agent_radius*2)

        self.boundary_endpts = [
            np.array([ 0, 0 ]),
            np.array([ self.WIDTH, 0 ]),
            np.array([ 0, self.HEIGHT ]),
            np.array([ self.WIDTH, self.HEIGHT ])
        ]
        self.boundary_endpts_wp = [endpt + self.window_pad for endpt in self.boundary_endpts]

        # Simulation parameters
        self.N = N
        self.T = T
        self.t = 0
        self.with_visualization = with_visualization
        if self.with_visualization:
            self.framerate_orig = framerate
        else:
            # this is more than what is possible with pygame so it will use the maximal framerate
            self.framerate_orig = 2000
        self.framerate = self.framerate_orig # distinguished for varying in-game framerate
        self.is_paused = False
        self.print_enabled = print_enabled
        self.plot_trajectory = plot_trajectory

        # Tracking parameters
        self.log_zarr_file = log_zarr_file

        if not self.log_zarr_file: # set up agent/resource data logging
            self.data_agent = np.zeros( (self.N, self.T, 4) ) # (pos_x, pos_y, mode, coll_res)
            self.data_res = []

        self.views = views
        self.goal_pos = np.array([0, self.HEIGHT/2])

        self.act = np.zeros([self.T-1])
        # # onehot
        # self.o_pre = np.zeros([self.T-1])
        # self.o_next = np.zeros([self.T-1])
        # vector
        o_len = vis_field_res*2
        self.o_pre = np.zeros([self.T-1, o_len])
        self.o_next = np.zeros([self.T-1, o_len])

        self.x_ag = x
        self.y_ag = y

        self.mode = mode
        self.start_dist = 0

        self.elapsed_time = 0
        # self.fitnesses = []
        self.save_ext = save_ext

        # Agent parameters
        self.agent_radii = agent_radius
        self.max_vel = max_vel
        self.vis_field_res = vis_field_res
        self.vision_range = vision_range
        self.agent_fov = agent_fov
        self.show_vision_range = show_vision_range
        self.agent_consumption = agent_consumption

        # Resource parameters
        self.N_res = N_res
        self.res_radius = patch_radius
        self.res_pos = res_pos
        self.min_res_units, self.max_res_units = res_units
        self.min_res_quality, self.max_res_quality = res_quality
        # fix units/quality to single values if not ranges
        if self.max_res_units <= self.min_res_units:
            self.max_res_units = self.min_res_units + 1 # randint is exclusive
        if self.max_res_quality < self.min_res_quality:
            self.max_res_quality = self.min_res_quality # uniform is inclusive
        self.regenerate_resources = regenerate_patches

        # Neural Network parameters
        self.model = NN

        if N == 1:  self.num_class_elements = 4 # single-agent --> perception of 4 walls
        else:       self.num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes
        # self.num_class_elements = 4

        self.other_input = other_input
        self.max_dist = np.hypot(self.WIDTH, self.HEIGHT)
        self.min_dist = agent_radius
        self.vis_transform = vis_transform
        self.percep_angle_noise_std = percep_angle_noise_std
        self.percep_dist_noise_std = percep_dist_noise_std
        self.action_noise_std = action_noise_std

        # Initializing pygame
        if self.with_visualization:
            pygame.init()
            self.screen = pygame.display.set_mode([self.WIDTH + self.window_pad*2, self.HEIGHT + self.window_pad*2])
            self.font = pygame.font.Font(None, int(self.window_pad/2))
            # self.recorder = ScreenRecorder(self.WIDTH + self.window_pad*2, self.HEIGHT + self.window_pad*2, framerate, out_file='sim.mp4')
        else:
            pygame.display.init()
            pygame.display.set_mode([1,1])

        # pygame related class attributes
        self.walls = pygame.sprite.Group()
        self.agents = pygame.sprite.Group()
        self.resources = pygame.sprite.Group()
        self.clock = pygame.time.Clock() # todo: look into this more in detail so we can control dt

### -------------------------- DRAWING FUNCTIONS -------------------------- ###

    def draw_walls(self):
        """Drawing walls on the arena according to initialization"""
        TL,TR,BL,BR = self.boundary_endpts_wp
        pygame.draw.line(self.screen, colors.BLACK, TL, TR)
        pygame.draw.line(self.screen, colors.BLACK, TR, BR)
        pygame.draw.line(self.screen, colors.BLACK, BR, BL)
        pygame.draw.line(self.screen, colors.BLACK, BL, TL)

    def draw_status(self):
        """Showing framerate, sim time and pause status on simulation windows"""
        status = [
            f"FPS: {self.framerate}  |  t = {self.t}/{self.T}",
        ]
        if self.is_paused:
            status.append("-Paused-")
        for i, stat_i in enumerate(status):
            text = self.font.render(stat_i, True, colors.BLACK)
            self.screen.blit(text, (self.window_pad, 0))

    def draw_agent_stats(self, font_size=15, spacing=0):
        """Showing agent information"""
        font = pygame.font.Font(None, font_size)
        for agent in self.agents:
            status = [
                f'ID: {agent.id}',
                f'res: {agent.collected_r}',
                f'ori: {agent.orientation*180/np.pi:.2f} deg',
                f'NNout: {agent.action:.2f}',
                f'turn: {agent.action*180/np.pi:.2f} deg',
                f'vel: {agent.velocity:.2f} / {self.max_vel}',
            ]
            for i, stat_i in enumerate(status):
                text = font.render(stat_i, True, colors.BLACK)
                self.screen.blit(text, (agent.position[0] + 8*agent.radius,
                                        agent.position[1] - 1*agent.radius + i * (font_size + spacing)))

    def draw_visual_fields(self):
        """Visualizing range of vision as opaque circles around the agents""" 
        vis_proj_distance = 30
        vis_project_IDbubble_size = 4
        
        for agent in self.agents:
            # Show visual range as circle if non-limiting FOV
            if self.agent_fov == 1:
                pygame.draw.circle(self.screen, colors.GREY, agent.pt_eye + self.window_pad, vis_proj_distance, width=1)
            else: # self.agent_fov < 1 --> show limits of FOV as radial lines with length of visual range
                start_pos = agent.pt_eye + self.window_pad
                angles = (agent.orientation + agent.phis[0], 
                          agent.orientation + agent.phis[-1])
                for angle in angles: ### draws lines that don't quite meet borders
                    end_pos = (start_pos[0] + np.cos(angle) * vis_proj_distance,
                               start_pos[1] - np.sin(angle) * vis_proj_distance)
                    pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)

            # draw projections as gray lines, either ending at walls (if dist_field is calculated) or extending beyond
            if self.vis_transform:

                for phi, vis_name, dist, dist_input in zip(agent.phis, agent.vis_field, agent.dist_field, agent.dist_input):

                    # draw lines to walls + bubble at wall end
                    end_pos = (start_pos[0] + np.cos(agent.orientation - phi) * dist,
                                start_pos[1] - np.sin(agent.orientation - phi) * dist)
                    pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)
                    pygame.draw.circle(self.screen, colors.BLACK, end_pos, 2)

                    # draw bubbles reflecting perceived identities (wall/agents) with radius proportional to dist_input
                    if vis_name == 'wall_north': # --> red
                        pygame.draw.circle(
                            self.screen, colors.TOMATO, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name == 'wall_south': # --> green
                        pygame.draw.circle(
                            self.screen, colors.LIME, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name == 'wall_east': # --> blue
                        pygame.draw.circle(
                            self.screen, colors.CORN, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name == 'wall_west': # --> yellow
                        pygame.draw.circle(
                            self.screen, colors.GOLD, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    elif vis_name == 'agent_exploit':
                        pygame.draw.circle(
                            self.screen, colors.VIOLET, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size)
                    else: # vis_name == 'agent_explore':
                        pygame.draw.circle(
                            self.screen, colors.BLACK, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = (dist_input+1)*vis_project_IDbubble_size*2)

            else:
                for phi, vis_name in zip(agent.phis, agent.vis_field):

                    # draw lines to walls
                    end_pos = (start_pos[0] + np.cos(agent.orientation - phi) * 1500,
                                start_pos[1] - np.sin(agent.orientation - phi) * 1500)
                    pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)

                    # draw bubbles reflecting perceived identities (wall/agents)
                    if vis_name == 'wall_north': # --> red
                        pygame.draw.circle(
                            self.screen, colors.TOMATO, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    elif vis_name == 'wall_south': # --> green
                        pygame.draw.circle(
                            self.screen, colors.LIME, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    elif vis_name == 'wall_east': # --> blue
                        pygame.draw.circle(
                            self.screen, colors.CORN, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    elif vis_name == 'wall_west': # --> yellow
                        pygame.draw.circle(
                            self.screen, colors.GOLD, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    elif vis_name == 'agent_exploit':
                        pygame.draw.circle(
                            self.screen, colors.VIOLET, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)
                    else: # vis_name == 'agent_explore':
                        pygame.draw.circle(
                            self.screen, colors.BLACK, 
                            (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                            start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                            radius = vis_project_IDbubble_size)

    # @timer
    def draw_frame(self):
        """Drawing environment, agents and every other visualization in each timestep"""
        pygame.display.flip()
        self.screen.fill(colors.WHITE)
        # pygame.draw.circle(self.screen, colors.BLACK, (700,425), 5)
        self.walls.draw(self.screen)
        self.resources.draw(self.screen)
        self.agents.draw(self.screen)
        self.draw_walls()
        self.draw_status()
        self.draw_agent_stats()

        # vision range + projection field
        if self.show_vision_range: 
            self.draw_visual_fields()
    
### -------------------------- WALL FUNCTIONS -------------------------- ###
    
    def create_walls(self):

        walls = [
            ('wall_north', (self.WIDTH, self.coll_boundary_thickness), np.array([ self.x_min, self.y_min ])),
            ('wall_south', (self.WIDTH, self.coll_boundary_thickness), np.array([ self.x_min, self.y_max - self.coll_boundary_thickness ])),
            ('wall_east', (self.coll_boundary_thickness, self.HEIGHT), np.array([ self.x_max - self.coll_boundary_thickness, self.y_min ])),
            ('wall_west', (self.coll_boundary_thickness, self.HEIGHT), np.array([ self.x_min, self.y_min ]))
        ]

        for id, size, position in walls:
            wall = Wall(
                id=id,
                size=size,
                position=position,
                window_pad=self.window_pad
            )
            self.walls.add(wall)

### -------------------------- AGENT FUNCTIONS -------------------------- ###

    # @timer
    def create_agents(self):
        """
        Instantiates agent objects according to simulation parameters
        Randomly initializes position (center within arena borders)
        Randomly initializes orientation (0 : right, pi/2 : up)
        Adds agent class to PyGame sprite group class (faster operations than lists)
        """
        x_min, x_max, y_min, y_max = self.boundary_info_coll

        if self.N == 1:
            colliding_resources = [0]
            retries = 0 
            while len(colliding_resources) > 0:

                x = np.random.randint(x_min, x_max)
                y = np.random.randint(y_min, y_max)
                # x,y = x_min, y_min

                if self.x_ag is not None:
                    x = self.x_ag
                    y = self.y_ag
                
                orient = np.random.uniform(0, 2 * np.pi)

                # x,y = 300,200
                # orient = 0

                agent = Agent(
                        id=0,
                        position=(x, y),
                        orientation=orient,
                        max_vel=self.max_vel,
                        FOV=self.agent_fov,
                        vision_range=self.vision_range,
                        num_class_elements=self.num_class_elements,
                        vis_field_res=self.vis_field_res,
                        consumption=self.agent_consumption,
                        model=self.model,
                        boundary_endpts=self.boundary_endpts,
                        window_pad=self.window_pad,
                        radius=self.agent_radii,
                        color=colors.BLUE,
                        vis_transform=self.vis_transform,
                        percep_angle_noise_std=self.percep_angle_noise_std,
                    )
                
                if self.mode == 'test':
                    colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)
                else:  
                    colliding_resources = []

                retries += 1
                if retries > 10: print(f'Retries > 10')
            self.agents.add(agent)

        else: # N > 1

            # starts = np.array([
            #     (100, 100, 5.33),
            #     (100, 300, 0),
            #     (100, 500, 0),
            #     (100, 700, 0),
            #     (100, 900, 0.66),
            #     (300, 100, 4.5),
            #     (500, 100, 4.5),
            #     (700, 100, 4.5),
            #     (900, 100, 3.66),
            #     (900, 300, 3),
            #     (900, 500, 3),
            #     (900, 700, 3),
            #     (900, 900, 2.33),
            #     (300, 900, 1.5),
            #     (500, 900, 1.5),
            #     (700, 900, 1.5),
            # ])

            # starts = np.array([
            #     (50,600,0),
            #     (600,800,3),
            #     (800,700,1.5),
            #     (800,100,3),
            # ])

            for i in range(self.N):

                colliding_resources = [0]
                colliding_agents = [0]

                retries = 0
                while len(colliding_resources) > 0 or len(colliding_agents) > 0:

                    x = np.random.randint(x_min, x_max)
                    y = np.random.randint(y_min, y_max)

                    # x = 950
                    # y = 50 + 100*i
                    
                    orient = np.random.uniform(0, 2 * np.pi)

                    # orient = 3

                    # x,y,orient = starts[i,:]

                    agent = Agent(
                            id=i,
                            position=(x, y),
                            orientation=orient,
                            max_vel=self.max_vel,
                            FOV=self.agent_fov,
                            vision_range=self.vision_range,
                            num_class_elements=self.num_class_elements,
                            vis_field_res=self.vis_field_res,
                            consumption=self.agent_consumption,
                            model=self.model,
                            boundary_endpts=self.boundary_endpts,
                            window_pad=self.window_pad,
                            radius=self.agent_radii,
                            color=colors.BLUE,
                            vis_transform=self.vis_transform,
                            percep_angle_noise_std=self.percep_angle_noise_std,
                        )
                    
                    colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)
                    colliding_agents = pygame.sprite.spritecollide(agent, self.agents, False, supcalc.within_group_collision)

                    retries += 1
                    if retries > 10: print(f'Retries > 10')
                self.agents.add(agent)

    # @timer
    def save_data_agent(self):
        """Tracks key variables (position, mode, resources collected) via array for current timestep"""
        for agent in self.agents:
            x, y = agent.position
            pos_x = x
            pos_y = self.y_max - y

            if agent.mode == 'explore': mode_num = 0
            elif agent.mode == 'exploit': mode_num = 1
            elif agent.mode == 'collide': mode_num = 2
            else: raise ValueError('Agent Mode not tracked')

            self.data_agent[agent.id, self.t, :] = np.array((pos_x, pos_y, mode_num, agent.collected_r))

### -------------------------- RESOURCE FUNCTIONS -------------------------- ###

    # @timer
    def create_resources(self):

        # creates single resource patch
        id = 0
        units = np.random.randint(self.min_res_units, self.max_res_units)
        quality = np.random.uniform(self.min_res_quality, self.max_res_quality)

        resource = Resource(id, self.res_radius, self.res_pos, units, quality)
        self.resources.add(resource)

        if not self.log_zarr_file: # save in sim instance
            x,y = resource.position
            pos_x = x
            pos_y = self.y_max - y
            self.data_res.append([pos_x, pos_y, self.res_radius])

    # def consume(self, agent):
    #     """Carry out agent-resource interactions (depletion, destroying, notifying)"""
    #     # Call resource agent is on
    #     resource = agent.res_to_be_consumed

    #     # Increment remaining resource quantity
    #     depl_units, destroy_res = resource.deplete(agent.consumption)

    #     # Update agent info
    #     if depl_units > 0:
    #         agent.collected_r += depl_units
    #         agent.mode = 'exploit'
    #     else:
    #         agent.mode = 'explore'

    #     # Kill + regenerate patch when fully depleted
    #     if destroy_res:
    #         resource.kill()
    #         if self.regenerate_resources:
    #             # self.add_new_resource_patch_random()
    #             self.add_new_resource_patch_stationary_single()

### -------------------------- COLLISION FUNCTIONS -------------------------- ###

    # @timer
    def collide_agent_res(self):

        # Create dict of every agent that has collided : [colliding resources]
        collision_group_ar = pygame.sprite.groupcollide(self.agents, self.resources, False, False, pygame.sprite.collide_circle)

        # Switch on all agents currently on a resource 
        for agent, resource_list in collision_group_ar.items():
            for resource in resource_list:
                # Flip bool variable if agent is within patch boundary
                if supcalc.distance(agent.position, resource.position) <= resource.radius:
                    agent.mode = 'exploit'
                    agent.on_res = 1
                    # agent.on_res_last_step = 1
                    agent.res_to_be_consumed = resource
                    break

    def collide_agent_wall(self):
        
        # Create dict of every agent that has collided : [colliding walls]
        collision_group_aw = pygame.sprite.groupcollide(self.agents, self.walls, False, False)

        # Change agent mode + note points of contact (carry out velocity-stopping check later in agent.move())
        for agent, wall_list in collision_group_aw.items():

            agent.mode = 'collide'

            for wall in wall_list:

                clip = agent.rect.clip(wall.rect)
                if self.with_visualization: pygame.draw.rect(self.screen, pygame.Color('red'), clip)

                # print(f'agent {agent.rect.center, agent.position} collided with {wall.id} @ {clip.center}')

                agent.collided_points.append(np.array(clip.center) - self.window_pad)

                # hits = [edge for edge in ['bottom', 'top', 'left', 'right'] if getattr(clip, edge) == getattr(agent.rect, edge)]
                # text = self.font.render(f'Collision at {", ".join(hits)}', True, pygame.Color('black'))
                # self.screen.blit(text, (self.window_pad, int(self.window_pad/2)))

    def collide_agent_agent(self):

        # Create dict of every agent that has collided : [colliding agents]
        collision_group_aa = pygame.sprite.groupcollide(self.agents, self.agents, False, False, supcalc.within_group_collision)
        
        # Carry out agent-agent collisions + generate list of collided agents
        for agent1, other_agents in collision_group_aa.items():

            agent1.mode = 'collide'

            for agentX in other_agents:

                clip = agent1.rect.clip(agentX.rect)
                if self.with_visualization: pygame.draw.rect(self.screen, pygame.Color('red'), clip)

                # print(f'agent {agent.rect.center} collided with {wall.id} @ {clip.center}')

                agent1.collided_points.append(np.array(clip.center) - self.window_pad)

### -------------------------- HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def interact_with_event(self, events):
        """Carry out functionality according to user's interaction"""
        for event in events:
            # Exit if requested
            if event.type == pygame.QUIT:
                sys.exit()

            # # Change orientation with mouse wheel
            # if event.type == pygame.MOUSEWHEEL:
            #     if event.y == -1:
            #         event.y = 0
            #     for ag in self.agents:
            #         ag.move_with_mouse(pygame.mouse.get_pos(), event.y, 1 - event.y)

            # Pause on Space
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.is_paused = not self.is_paused

            # Speed up on s and down on f. reset default framerate with d
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.framerate -= 1
                if self.framerate < 1:
                    self.framerate = 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.framerate += 1
                if self.framerate > 100:
                    self.framerate = 100
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.framerate = self.framerate_orig

            # # Continuous mouse events (move with cursor)
            # if pygame.mouse.get_pressed()[0]:
            #     try:
            #         for ag in self.agents:
            #             ag.move_with_mouse(event.pos, 0, 0)
            #         for res in self.resources:
            #             res.update_clicked_status(event.pos)
            #     except AttributeError:
            #         for ag in self.agents:
            #             ag.move_with_mouse(pygame.mouse.get_pos(), 0, 0)
            # else:
            #     for ag in self.agents:
            #         ag.is_moved_with_cursor = False
            #         ag.draw_update()
            #     for res in self.resources:
            #         res.is_clicked = False
            #         res.draw_update()


##################################################################################
### -------------------------- MAIN SIMULATION LOOP -------------------------- ###
##################################################################################

    def start(self):

        ### ---- INITIALIZATION ---- ###

        start_time = time.time()
        self.create_walls()
        self.create_resources()
        self.create_agents()

        # obs_times = np.zeros(self.T)
        # mod_times = np.zeros(self.T)
        # sav_times = np.zeros(self.T)
        # ful_times = np.zeros(self.T)

        ### ---- START OF SIMULATION ---- ###

        while self.t < self.T:

            if not self.is_paused:

                # self.recorder.capture_frame(self.screen)
                
                ### ---- OBSERVATIONS ---- ###

                # obs_start = time.time()

                # Refresh agent behavioral states
                for agent in self.agents:
                    
                    agent.collided_points = []
                    agent.mode = 'explore'
                    agent.on_res = 0
                    # if agent.on_res_last_step > 0: # 1 timestep memory
                    #     agent.on_res_last_step = 0
                    # elif agent.on_res > 0:
                    #     agent.on_res = 0

                # Evaluate sprite interactions + flip agent modes to 'collide'/'exploit' (latter takes precedence)
                self.collide_agent_wall()
                self.collide_agent_agent()
                self.collide_agent_res()

                # for res in self.resources:
                #     res_pos = res.position
                for ag in self.agents:
                    ag_pos = ag.position
                # self.start_dist = np.linalg.norm(res_pos - ag_pos)
                self.goal_dist = np.linalg.norm(self.goal_pos - ag_pos)

                # Update visual projections
                for agent in self.agents:
                    agent.visual_sensing(self.agents)

                # obs_times[self.t] = time.time() - obs_start

                ### ---- VISUALIZATION ---- ###

                if self.with_visualization:
                    for agent in self.agents:
                        agent.draw_update() 
                    for res in self.resources:
                        res.draw_update() 
                    self.draw_frame()
                else: # still have to update rect for collisions
                    for agent in self.agents:
                        agent.rect = agent.image.get_rect(center = agent.position + self.window_pad)

                ### ---- TRACKING ---- ### 

                # sav_start = time.time()
                if self.log_zarr_file:
                    tracking.save_agent_data_RAM(self)
                    tracking.save_resource_data_RAM(self)
                else:
                    self.save_data_agent()
                # sav_times[self.t] = time.time() - sav_start

                ### ---- MODEL + ACTIONS ---- ###

                # mod_start = time.time()
                for agent in self.agents:

                    # Observe + encode sensory inputs
                    vis_input = agent.encode_one_hot(agent.vis_field)
                    agent.dist_input = np.array(agent.dist_field)

                    if self.vis_transform != '':
                        if self.vis_transform == 'minmax':
                            agent.dist_input = (agent.dist_input - self.min_dist)/(self.max_dist - self.min_dist) # flipped --> todo: switch + scale
                        elif self.vis_transform == 'far':
                            agent.dist_input = self.min_dist*2 / agent.dist_input # holdover ---> change if using
                        # elif self.vis_transform == 'far':
                        #     dist_input = ( self.min_dist / dist_input + .25 ) / 1.25
                        # elif self.vis_transform == 'mfar':
                        #     dist_input = ( np.power(self.min_dist / dist_input, 1.2) + .25 ) / 1.25
                        # elif self.vis_transform == 'vfar':
                        #     dist_input = ( np.power(self.min_dist / dist_input, 2) + .25 ) / 1.25
                        elif self.vis_transform == 'maxWF':
                            agent.dist_input = 1.465 - np.log(agent.dist_input) / 5 # bounds [min, max] within [0, 1]
                        elif self.vis_transform == 'WF':
                            agent.dist_input = 1.24 - np.log(agent.dist_input) / 7 # bounds [min, max] within [0.2, 0.9]
                        elif self.vis_transform == 'mlWF':
                            agent.dist_input = 1 - np.log(agent.dist_input) / 9.65 # bounds [min, max] within [0.25, 0.75]
                        elif self.vis_transform == 'mWF':
                            agent.dist_input = .9 - np.log(agent.dist_input) / 12 # bounds [min, max] within [0.3, 0.7]
                        elif self.vis_transform == 'msWF':
                            agent.dist_input = .8 - np.log(agent.dist_input) / 16 # bounds [min, max] within [0.35, 0.65]
                        elif self.vis_transform == 'sWF':
                            agent.dist_input = .7 - np.log(agent.dist_input) / 24 # bounds [min, max] within [0.4, 0.6]
                        # elif self.vis_transform == 'minmax_buffer':
                        #     dist_input = (dist_input - self.min_dist + 100)/(self.max_dist - self.min_dist + 200)
                        #     dist_input *= np.abs(np.random.randn(dist_input.shape[0]) * self.sensory_noise_std + 1)
                        #     vis_input *= dist_input
                        # elif self.vis_transform == 'minmax_scalp1':
                        #     dist_input = (dist_input - self.min_dist)/(self.max_dist - self.min_dist)
                        #     dist_input *= np.abs(np.random.randn(dist_input.shape[0]) * self.sensory_noise_std + 1)
                        #     vis_input -= dist_input*.1
                        # elif self.vis_transform == 'minmax_scalp5':
                        #     dist_input = (dist_input - self.min_dist)/(self.max_dist - self.min_dist)
                        #     dist_input *= np.abs(np.random.randn(dist_input.shape[0]) * self.sensory_noise_std + 1)
                        #     vis_input -= dist_input*.5

                        # add noise + perturbation + clip
                        # agent.dist_input += np.random.randn(agent.dist_input.shape[0]) * self.percep_dist_noise_std
                        # dist_input *= 1.1
                        # dist_input /= 1.5
                        # dist_input += .05
                        agent.dist_input = np.clip(agent.dist_input, 0,1)

                        vis_input *= agent.dist_input

                    # print(np.round(np.sum(vis_input,axis=0),4))

                    # if agent.mode == 'collide': other_input = np.array([agent.on_res, 1])
                    # else:                       other_input = np.array([agent.on_res, 0])

                    # Calculate action
                    # agent.action = agent.model.forward(vis_input, np.array([agent.on_res]), agent.hidden)

                    # if agent.on_res == 1:
                    #     print('on')

                    # # onehot
                    # o_current = self.views.index(agent.vis_field) + agent.on_res*len(self.views)
                    # # o_current = self.views.index(agent.vis_field)
                    # flat matrix
                    # o_current = vis_input.flatten()
                    # vector
                    vis_by_ray = agent.encode_one_hot_by_ray(agent.vis_field)
                    o_current = np.hstack((vis_by_ray, agent.dist_input))
                    # print(o_current)

                    # print(agent.on_res, o_idx)
                    # print(vis_input)

                    if self.mode=='train':
                        act_idx, agent.action = agent.model.gaussian_action()
                    elif self.mode=='test':
                        act_idx, agent.action = agent.model.planned_action(o_current)

                    if self.t < self.T-1:
                        # self.o_pre[self.t] = o_current # onehot
                        self.o_pre[self.t,:] = o_current # vector
                        self.act[self.t] = act_idx
                    if self.t > 0:
                        # self.o_next[self.t-1] = o_current # onehot
                        self.o_next[self.t-1,:] = o_current # vector

                    # if o_current == agent.model.goal and self.mode == 'test': # onehot
                    # if self.mode == 'test' and np.all(np.isclose(o_current, agent.model.goal, atol=5e-02)): # vector, dist
                    if self.mode == 'test' and agent.mode == 'exploit': # foor present

                            # self.recorder.end_recording()
                            pygame.quit()
                            return (self.o_pre, self.act, self.o_next), self.t, self.goal_dist

                    else: # No food (or training) --> move (stay stationary if collided object in front)
                        action = agent.action + np.random.randn()*self.action_noise_std
                        agent.move(action)
                        # agent.move(0.1)
                        # agent.move(np.random.uniform(-0.1,0.1))

                # mod_times[self.t] = time.time() - mod_start

            ### ---- BACKGROUND PROCESSES ---- ###
        
                # Step sim time forward
                self.t += 1

                # Step clock time to calculate fps
                if self.with_visualization:
                    self.clock.tick(self.framerate)
                    if self.print_enabled and (self.t % 500 == 0):
                        print(f"t={self.t} \t| FPS: {round(self.clock.get_fps(),1)}")
                
                # ful_times[self.t-1] = time.time() - obs_start

            # Carry out user interactions even when not paused
            if self.with_visualization:
                events = pygame.event.get() 
                self.interact_with_event(events)

        # self.recorder.end_recording()
        pygame.quit()
        return (self.o_pre, self.act, self.o_next), self.t, self.goal_dist