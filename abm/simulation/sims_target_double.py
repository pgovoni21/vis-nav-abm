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
    def __init__(self, width=600, height=480, window_pad=30,
                 N=1, T=1000, with_visualization=True, framerate=25, print_enabled=False, 
                 plot_trajectory=False, log_zarr_file=False, save_ext="",
                 agent_radius=10, max_vel=5, vis_field_res=8, vision_range=150, agent_fov=1.0, 
                 visual_exclusion=False, show_vision_range=False, agent_consumption=1, 
                 N_resrc=10, patch_radius=30, min_resrc_perpatch=200, max_resrc_perpatch=1000, 
                 min_resrc_quality=0.1, max_resrc_quality=1, regenerate_patches=True, 
                 NN=None, RNN_other_input_size=1, CNN_depths=[1,], CNN_dims=[4,], RNN_hidden_size=128, LCL_output_size=1, 
                 NN_activ='relu', RNN_type='fnn',
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
        :param visual_exclusion: when true agents can visually exclude socially relevant visual cues from other agents'
                                projection field
        :param show_vision_range: bool to switch visualization of visual range for agents. If true the limit of far
                                and near field visual field will be drawn around the agents
        :param agent_consumption: agent consumption (exploitation speed) in res. units / time units
        :param N_resrc: number of resource patches in the environment
        :param patch_radius: radius of resrcaurce patches
        :param min_resrc_perpatch: minimum resource unit per patch
        :param max_resrc_perpatch: maximum resource units per patch
        :param min_resrc_quality: minimum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        : param max_resrc_quality: maximum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        :param regenerate_patches: bool to decide if patches shall be regenerated after depletion
        :param NN:
        :param NN_input_other_size:
        :param NN_hidden_size:
        :param NN_output_size:
        :param NN_activ:
        """
        # Arena parameters
        self.WIDTH = width
        self.HEIGHT = height
        self.window_pad = window_pad
        self.coll_boundary_thickness = agent_radius

        self.x_min, self.x_max = 0, width
        self.y_min, self.y_max = 0, height
        
        self.boundary_info = (0, width, 0, height)
        self.boundary_endpts = [
            np.array([ 0, 0 ]),
            np.array([ width, 0 ]),
            np.array([ 0, height ]),
            np.array([ width, height ])
        ]
        self.boundary_endpts_wp = [endpt + self.window_pad for endpt in self.boundary_endpts]

        self.boundary_info_spwn = (width*.37, width*.63, height*.37, height*.63)
        self.spwn_endpts = [
            np.array([ width*.37, height*.37 ]),
            np.array([ width*.63, height*.37 ]),
            np.array([ width*.37, height*.63 ]),
            np.array([ width*.63, height*.63 ])
        ]
        self.spwn_endpts_wp = [endpt + self.window_pad for endpt in self.spwn_endpts]

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

        self.elapsed_time = 0
        self.fitnesses = []
        self.save_ext = save_ext

        # Agent parameters
        self.agent_radii = agent_radius
        self.max_vel = max_vel
        self.vis_field_res = vis_field_res
        self.vision_range = vision_range
        self.agent_fov = agent_fov
        self.visual_exclusion = visual_exclusion
        self.show_vision_range = show_vision_range
        self.agent_consumption = agent_consumption

        # Resource parameters
        self.N_resrc = N_resrc
        self.resrc_radius = patch_radius
        self.min_resrc_units = min_resrc_perpatch
        self.max_resrc_units = max_resrc_perpatch
        self.min_resrc_quality = min_resrc_quality
        self.max_resrc_quality = max_resrc_quality
        # possibility to provide single values instead of value ranges
        # if maximum values are negative for both quality and contained units
        if self.max_resrc_quality < 0:
            self.max_resrc_quality = self.min_resrc_quality
        if self.max_resrc_units < 0:
            self.max_resrc_units = self.min_resrc_units + 1
        self.regenerate_resources = regenerate_patches
        self.res_id_og = np.random.randint(2)
        self.res_id_counter = self.res_id_og

        # Neural Network parameters
        self.model = NN

        if N == 1:  self.num_class_elements = 4 # single-agent --> perception of 4 walls
        else:       self.num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes
        
        CNN_input_size = (self.num_class_elements, vis_field_res)
        self.architecture = (
            CNN_input_size, 
            CNN_depths, 
            CNN_dims, 
            RNN_other_input_size, 
            RNN_hidden_size, 
            LCL_output_size
            )
        self.NN_activ = NN_activ
        self.RNN_type = RNN_type

        # if print_enabled: 
        #     print(f"NN inputs = {self.vis_size} (vis_size) + {self.contact_size} (contact_size)",end="")
        #     print(f" + {self.other_size} (velocity + orientation) = {self.NN_input_size}")
        #     print(f"Agent NN architecture = {(self.NN_input_size, NN_hidden_size, self.NN_output_size)}")
        #     print(f"NN_activ: {NN_activ}")

        # Initializing pygame
        if self.with_visualization:
            pygame.init()
            self.screen = pygame.display.set_mode([self.WIDTH + self.window_pad*2, self.HEIGHT + self.window_pad*2])
            self.font = pygame.font.Font(None, int(self.window_pad/2))
            # self.recorder = ScreenRecorder(self.x_min + self.x_max, self.y_min + self.y_max, framerate, out_file='sim.mp4')
        else:
            # pass
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

        TL,TR,BL,BR = self.spwn_endpts_wp
        pygame.draw.line(self.screen, colors.GREY, TL, TR)
        pygame.draw.line(self.screen, colors.GREY, TR, BR)
        pygame.draw.line(self.screen, colors.GREY, BR, BL)
        pygame.draw.line(self.screen, colors.GREY, BL, TL)

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
        vis_project_IDbubble_size = 2
        
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

            # for each visual field ray
            for phi, vis_name in zip(agent.phis, agent.vis_field):

                # # draw projections as gray lines
                # end_pos = (start_pos[0] + np.cos(agent.orientation - phi) * 1500,
                #             start_pos[1] - np.sin(agent.orientation - phi) * 1500)
                # pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)

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
        self.walls.draw(self.screen)
        self.resources.draw(self.screen)
        self.agents.draw(self.screen)
        self.draw_walls()
        self.draw_status()
        # self.draw_agent_stats()

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
        x_min, x_max, y_min, y_max = self.boundary_info_spwn

        if self.N == 1:
            colliding_resources = [0]
            retries = 0
            while len(colliding_resources) > 0:

                x = np.random.randint(x_min, x_max)
                y = np.random.randint(y_min, y_max)
                # x,y = 981, 20
                
                orient = np.random.uniform(0, 2 * np.pi)
                # orient = .5

                agent = Agent(
                        id=0,
                        position=(x, y),
                        orientation=orient,
                        max_vel=self.max_vel,
                        FOV=self.agent_fov,
                        vision_range=self.vision_range,
                        visual_exclusion=self.visual_exclusion,
                        consumption=self.agent_consumption,
                        arch=self.architecture,
                        model=self.model,
                        NN_activ = self.NN_activ,
                        RNN_type = self.RNN_type,
                        boundary_endpts=self.boundary_endpts,
                        radius=self.agent_radii,
                        color=colors.BLUE,
                    )
                
                colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)

                retries += 1
                if retries > 10: print(f'Retries > 10')
            self.agents.add(agent)

        else: # N > 1
            for i in range(self.N):

                colliding_resources = [0]
                colliding_agents = [0]

                retries = 0
                while len(colliding_resources) > 0 or len(colliding_agents) > 0:

                    x = np.random.randint(x_min, x_max)
                    y = np.random.randint(y_min, y_max)
                    # x,y = 305+15*i, 305+15*i
                    
                    orient = np.random.uniform(0, 2 * np.pi)

                    agent = Agent(
                            id=i,
                            position=(x, y),
                            orientation=orient,
                            max_vel=self.max_vel,
                            FOV=self.agent_fov,
                            vision_range=self.vision_range,
                            visual_exclusion=self.visual_exclusion,
                            consumption=self.agent_consumption,
                            arch=self.architecture,
                            model=self.model,
                            NN_activ = self.NN_activ,
                            RNN_type = self.RNN_type,
                            boundary_endpts=self.boundary_endpts,
                            radius=self.agent_radii,
                            color=colors.BLUE,
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

        # creates single resource patch in alternating positions
        id = self.res_id_counter

        # top-left / bottom-right corners
        self.resrc_radius = self.WIDTH*.1 # 100 if width=1000
        x_min, x_max, y_min, y_max = self.boundary_info_spwn

        if self.res_id_counter % 2 == 0: 
            x,y = x_min,y_min
        else:
            x,y = x_max,y_max

        # # top-left / bottom-right points, off-center / off-wall / asym (centers @ 140,450 - 450,140)
        # self.resrc_radius = 20
        # if self.res_id_counter % 2 == 0:
        #     x = self.x_min + 120
        #     y = self.y_min + 30
        # else:
        #     x = self.x_max - self.resrc_radius/2 - self.window_pad - 30
        #     y = self.y_max - self.resrc_radius/2 - self.window_pad - 120

        # # top-left / bottom-right points, off-center / off-wall / sym (centers @ 140,450 - 360,50)
        # self.resrc_radius = 20
        # if self.res_id_counter % 2 == 0:
        #     x = self.x_min + 120
        #     y = self.y_min + 30
        # else:
        #     x = self.x_max - self.resrc_radius/2 - self.window_pad - 120
        #     y = self.y_max - self.resrc_radius/2 - self.window_pad - 30

        units = np.random.randint(self.min_resrc_units, self.max_resrc_units)
        quality = np.random.uniform(self.min_resrc_quality, self.max_resrc_quality)

        save_id = id - self.res_id_og # to start tracking matrix at zero
        resource = Resource(save_id, self.resrc_radius, (x, y), units, quality)
        self.resources.add(resource)

        if not self.log_zarr_file: # save in sim instance
            x,y = resource.position
            pos_x = x
            pos_y = self.y_max - y
            self.data_res.append([pos_x, pos_y, self.resrc_radius])

    # @timer
    def consume(self, agent):
        """Carry out agent-resource interactions (depletion, destroying, notifying)"""
        # Call resource agent is on
        resource = agent.res_to_be_consumed

        # Increment remaining resource quantity
        depl_units, destroy_resrc = resource.deplete(agent.consumption)

        # Update agent info
        if depl_units > 0:
            agent.collected_r += depl_units
            agent.mode = 'exploit'
        else:
            agent.mode = 'explore'

        # Kill + regenerate patch when fully depleted
        if destroy_resrc:
            resource.kill()
            if self.regenerate_resources:
                self.res_id_counter += 1 # alternates res position
                self.create_resources()

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
                    agent.on_resrc = 1
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

                # print(f'agent {agent.rect.center} collided with {wall.id} @ {clip.center}')

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
                    if agent.on_resrc > 0:
                        agent.on_resrc -= 0.2 # fades over 5 timesteps
                        agent.on_resrc = round(agent.on_resrc,1)

                # Evaluate sprite collisions + flip agent modes to 'collide'/'exploit' (latter takes precedence)
                self.collide_agent_wall()
                self.collide_agent_agent()
                self.collide_agent_res()

                # Update visual projections (vis_field)
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
                else: # still have to update rect for wall collisions
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

                    # Pass observables through NN to calculate action & advance agent's hidden state
                    vis_input = agent.encode_one_hot(agent.vis_field)
                    agent.action, agent.hidden = agent.model.forward(vis_input, np.array([agent.on_resrc]), agent.hidden)

                    # Food present --> consume (if food is still available)
                    if agent.mode == 'exploit':
                        self.consume(agent) 
                    # No food --> move via decided action (stationary if collided object in front)
                    else: 
                        agent.move(agent.action)
                        # agent.move(0.1)
                        # agent.move(np.random.uniform(-0.1,0.1))

                # mod_times[self.t] = time.time() - mod_start

            ### ---- BACKGROUND PROCESSES ---- ###
        
                # Step simulation time forward
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

        ### ---- END OF SIMULATION ---- ###

        # self.recorder.end_recording()
        pygame.quit()

        # compute simulation time in seconds
        self.elapsed_time = round( (time.time() - start_time) , 2)
        if self.print_enabled:
            print(f"Elapsed_time: {self.elapsed_time}")

        if self.log_zarr_file:
            # conclude agent/resource tracking
            # convert tracking agent/resource dicts to N-dimensional zarr arrays + save to offline file
            ag_zarr, res_zarr = tracking.save_zarr_file(self.T, self.save_ext, self.print_enabled)
            plot_data = ag_zarr, res_zarr
            # extract total fitnesses of each agent + save into sim instance (pulled for EA)
            self.fitnesses = ag_zarr[:,-1,-1]
        else: # use ag/res tracking from self instance
            # convert list to 3D array similar to zarr file
            data_res_array = np.zeros( (len(self.data_res), 1, 3 ))
            for id, (pos_x, pos_y, radius) in enumerate(self.data_res):
                data_res_array[id, 0, 0] = pos_x
                data_res_array[id, 0, 1] = pos_y
                data_res_array[id, 0, 2] = radius

            # assign plot data as numpy arrays
            plot_data = self.data_agent, data_res_array
            # extract agent fitnesses from self
            self.fitnesses = self.data_agent[:,-1,-1]

        # print list of agent fitnesses
        if self.print_enabled:
            print(f"Fitnesses: {self.fitnesses}")

        # display static map of simulation
        if self.plot_trajectory:
            plot_funcs.plot_map(plot_data, self.WIDTH, self.HEIGHT, self.coll_boundary_thickness, save_name=self.save_ext)

        return self.fitnesses, self.elapsed_time