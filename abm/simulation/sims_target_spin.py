import pygame
import numpy as np
import sys
import time

from abm import colors
from abm.sprites import supcalc
from abm.sprites.agent import Agent
from abm.sprites.resource import Resource
from abm.sprites.wall import Wall
# from abm.monitoring.screen_recorder import ScreenRecorder
# from abm.helpers import timer

class Simulation:
    # @timer
    def __init__(self, env_size, window_pad,
                 N, N_rand, T, with_visualization, framerate, save_ext,
                 agent_radius, max_vel, vis_field_res, vision_range, agent_fov, show_vision_range, agent_consumption, agent_collide, agent_patch_collide,
                 N_res, patch_radius, res_pos, res_units, res_quality, regenerate_patches, 
                 NN, other_input, vis_transform, percep_angle_noise_std, percep_dist_noise_std, action_noise_std,
                 boundary_scale, sim_type, RW_rot_diff, social_init_range, social_init_type, init_info=None, feat_out=False,
                 ):

        # Arena parameters
        self.WIDTH, self.HEIGHT = env_size
        self.window_pad = window_pad
        self.coll_boundary_thickness = agent_radius

        self.x_min, self.x_max = 0, self.WIDTH
        self.y_min, self.y_max = 0, self.HEIGHT
        # self.boundary_info_coll = (agent_radius*2, self.WIDTH - agent_radius*2, 
        #                            agent_radius*2, self.HEIGHT - agent_radius*2)

        self.boundary_endpts = [
            np.array([ -boundary_scale, -boundary_scale ]),
            np.array([ self.WIDTH+boundary_scale, -boundary_scale ]),
            np.array([ -boundary_scale, self.HEIGHT+boundary_scale ]),
            np.array([ self.WIDTH+boundary_scale, self.HEIGHT+boundary_scale ])
        ]
        self.boundary_endpts_wp = [endpt + self.window_pad for endpt in self.boundary_endpts]

        # Simulation parameters
        self.N = N
        self.N_rand = N_rand
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
        self.sim_type = sim_type

        # Agent parameters
        self.respawn_counter = 0
        self.respawn_times = []
        self.agent_radii = agent_radius
        self.max_vel = max_vel
        self.vis_field_res = vis_field_res
        self.vision_range = vision_range
        self.agent_fov = agent_fov
        self.show_vision_range = show_vision_range
        self.agent_consumption = agent_consumption
        self.agent_collide = agent_collide
        self.agent_patch_collide = agent_patch_collide
        self.social_init_range = social_init_range
        self.social_init_type = social_init_type

        # Params from build_agent_trajs()
        # if init_info is not None:
        #     x,y,orient,timesteps,perturb = init_info
        z,spin_angle,timesteps,perturb = init_info
        self.T = timesteps # override envconf
        self.agent_init = z # pass to create_agent
        self.spin_angle = spin_angle # controls agent.move()
        # else:
        #     self.agent_init = None

        # Tracking parameters
        self.feat_out = feat_out
        if self.feat_out:
            self.data_agent = np.zeros( (self.T, 24) ) # (pos_x, pos_y, ori, action, vis_feat, RNN_out)
        else:
            self.data_agent = np.zeros( (self.T, 4) ) # (pos_x, pos_y, ori, action)
        self.data_res = []
        self.vis_feat = np.zeros(4)
        self.RNN_out = np.zeros(16)
        self.save_ext = save_ext

        # Boundary-Ray Collision parameters
        phis = np.linspace(-agent_fov*np.pi, agent_fov*np.pi, vis_field_res)
        self.phi_angle_diff = phis[1] - phis[0]
        self.fwd_traj = np.array([[0,0],[0,0]])
        self.last_moves = []
        self.ellipse_counter = 0
        self.single_ray_colls = []
        self.dual_ray_colls = []
        self.colls_no_ellipse = []

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
        self.rot_diff = RW_rot_diff

        self.num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes

        self.other_input = other_input
        self.max_dist = np.hypot(self.WIDTH, self.HEIGHT)
        self.min_dist = agent_radius
        self.vis_transform = vis_transform
        self.percep_angle_noise_std = percep_angle_noise_std*2*np.pi # noise std as percentage * range
        self.percep_dist_noise_std = percep_dist_noise_std
        self.action_noise_std = action_noise_std*2

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
        self.objs = pygame.sprite.Group()
        self.agents = pygame.sprite.Group()
        if 'ghost' in self.sim_type:
            self.ghost = pygame.sprite.Group()
            self.viewable_agents = pygame.sprite.Group()
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
    
    def draw_objs(self):
        for obj in self.objs:
            pygame.draw.circle(self.screen, obj.color, obj.position + self.window_pad, obj.radius)

    def draw_status(self):
        """Showing framerate, sim time and pause status on simulation windows"""
        status = [
            # f"FPS: {self.framerate}  |  t = {self.t}/{self.T}",
            f"t = {self.t}/{self.T}",
        ]
        if self.is_paused:
            status.append("-Paused-")
        for i, stat_i in enumerate(status):
            text = self.font.render(stat_i, True, colors.BLACK)
            self.screen.blit(text, (self.window_pad, self.window_pad - self.agent_radii))

    def draw_agent_stats(self, font_size=15, spacing=0):
        """Showing agent information"""
        font = pygame.font.Font(None, font_size)
        for agent in self.agents:
            status = [ 
                f'ori: {int(agent.orientation*180/np.pi)} deg',
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

        # only for ANN agent
        agent = self.agents.sprites()[0]

        start_pos = agent.pt_eye + self.window_pad
        # Show visual range as circle if non-limiting FOV
        if self.agent_fov == 1:
            pygame.draw.circle(self.screen, colors.GREY, agent.pt_eye + self.window_pad, vis_proj_distance, width=1)
        else: # self.agent_fov < 1 --> show limits of FOV as radial lines with length of visual range
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
                elif vis_name == 'agent_explore':
                    pygame.draw.circle(
                        self.screen, colors.BLACK, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size+1) # edge
                    pygame.draw.circle(
                        self.screen, colors.CYAN, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size)
                elif vis_name == 'agent_exploit':
                    pygame.draw.circle(
                        self.screen, colors.BLACK, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size+1) # edge
                    pygame.draw.circle(
                        self.screen, colors.VIOLET, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size)

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
                elif vis_name == 'agent_explore':
                    pygame.draw.circle(
                        self.screen, colors.BLACK, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size+1) # edge
                    pygame.draw.circle(
                        self.screen, colors.CYAN, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size)
                elif vis_name == 'agent_exploit':
                    pygame.draw.circle(
                        self.screen, colors.BLACK, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size+1) # edge
                    pygame.draw.circle(
                        self.screen, colors.VIOLET, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * vis_proj_distance,
                        start_pos[1] - np.sin(agent.orientation - phi) * vis_proj_distance),
                        radius = vis_project_IDbubble_size)

        # # draw line to patch
        # end_pos = np.array(self.res_pos)
        # length = 100

        # # line from agent to corner
        # end_pos = np.array([0,0])
        # len = 500
        # # res_pos[1] = self.y_max - res_pos[1]
        # disp_from_patch = end_pos - agent.position
        # angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
        # angle_diff = angle_to_patch - agent.orientation
        # angle_diff = (angle_diff - np.pi) % (2*np.pi) - np.pi
        # # print(angle_to_patch, angle_diff)
        # end_pos = (start_pos[0] + np.cos(angle_to_patch)*len,
        #            start_pos[1] - np.sin(angle_to_patch)*len)
        # pygame.draw.line(self.screen, colors.BLACK, start_pos, end_pos, 1)

        # # line from agent to corner
        # end_pos = np.array([0,1000])
        # len = 500
        # # res_pos[1] = self.y_max - res_pos[1]
        # disp_from_patch = end_pos - agent.position
        # angle_to_patch = np.arctan2(-disp_from_patch[1], disp_from_patch[0])
        # angle_diff = angle_to_patch - agent.orientation
        # angle_diff = (angle_diff - np.pi) % (2*np.pi) - np.pi
        # # print(angle_to_patch, angle_diff)
        # end_pos = (start_pos[0] + np.cos(angle_to_patch)*len,
        #            start_pos[1] - np.sin(angle_to_patch)*len)
        # pygame.draw.line(self.screen, colors.BLACK, start_pos, end_pos, 1)

        # agent traveled traj
        for t_step in range(1,self.t):

            if t_step-1 in self.respawn_times:
                continue

            start = self.data_agent[t_step-1, :2]
            end = self.data_agent[t_step, :2]

            pygame.draw.line(self.screen, colors.BLACK, start + self.window_pad, end + self.window_pad, 1)

    # @timer
    def draw_frame(self):
        """Drawing environment, agents and every other visualization in each timestep"""
        pygame.display.flip()
        self.screen.fill(colors.WHITE)
        # pygame.draw.circle(self.screen, colors.BLACK, (700,425), 5)
        self.walls.draw(self.screen)
        self.resources.draw(self.screen)
        if 'ghost' in self.sim_type:
            self.viewable_agents.draw(self.screen)
        else:
            self.agents.draw(self.screen)
        self.draw_walls()
        self.draw_objs()
        self.draw_status()
        # self.draw_agent_stats()

        # vision range + projection field
        if self.show_vision_range: 
            self.draw_visual_fields()
    
# ### -------------------------- ENV FUNCTIONS -------------------------- ###
    
#     def create_walls(self):

#         walls = [
#             ('wall_north', (self.WIDTH, self.coll_boundary_thickness), np.array([ self.x_min, self.y_min ])),
#             ('wall_south', (self.WIDTH, self.coll_boundary_thickness), np.array([ self.x_min, self.y_max - self.coll_boundary_thickness ])),
#             ('wall_east', (self.coll_boundary_thickness, self.HEIGHT), np.array([ self.x_max - self.coll_boundary_thickness, self.y_min ])),
#             ('wall_west', (self.coll_boundary_thickness, self.HEIGHT), np.array([ self.x_min, self.y_min ]))
#         ]

#         for id, size, position in walls:
#             wall = Wall(
#                 id=id,
#                 size=size,
#                 position=position,
#                 window_pad=self.window_pad
#             )
#             self.walls.add(wall)

### -------------------------- AGENT FUNCTIONS -------------------------- ###

    # @timer
    def create_agents(self):
        """
        Instantiates agent objects according to simulation parameters
        Randomly initializes position (center within arena borders)
        Randomly initializes orientation (0 : right, pi/2 : up)
        Adds agent class to PyGame sprite group class (faster operations than lists)
        """
        # x_min, x_max, y_min, y_max = self.boundary_info_coll

        # ANN controlled agent

        # colliding_resources = [0]
        # retries = 0 
        # while len(colliding_resources) > 0:

            # if self.agent_init is None:
            #     x = np.random.randint(x_min, x_max)
            #     y = np.random.randint(y_min, y_max)            
            #     orient = np.random.uniform(0, 2 * np.pi)
            # else: # no-res traj sim
            #     x,y,orient = self.agent_init

        x,y = 1015,1015
        orient = np.pi/4

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
                sim_type=self.sim_type
            )
            
            # colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)

            # retries += 1
            # if retries > 10: print(f'Retries > 10')
        self.agents.add(agent)
        if 'ghost' in self.sim_type:
            self.viewable_agents.add(agent)

        # RW/D agents

        # if self.social_init_type == '':

        for i in range(1, self.N):

            # colliding_resources = [0]
            # colliding_agents = [0]

            # retries = 0
            # while len(colliding_resources) > 0 or len(colliding_agents) > 0:

                # x = np.random.randint(x_min, x_max)
                # y = np.random.randint(y_min, y_max)                
                # orient = np.random.uniform(0, 2 * np.pi)
            z = self.agent_init
            x,y,orient = 1000-z, 1000-z, 0

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
                    model=None,
                    boundary_endpts=self.boundary_endpts,
                    window_pad=self.window_pad,
                    radius=self.agent_radii,
                    color=colors.CYAN,
                    vis_transform=self.vis_transform,
                    percep_angle_noise_std=self.percep_angle_noise_std,
                    sim_type=self.sim_type
                )
                
                # colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)
                # if self.agent_collide:
                #     colliding_agents = pygame.sprite.spritecollide(agent, self.agents, False, supcalc.within_group_collision)
                # else:
                #     colliding_agents = []

                # retries += 1
                # if retries > 10: print(f'Retries > 10')
            self.agents.add(agent)
            if 'ghost' in self.sim_type:
                self.viewable_agents.add(agent)
        
        # else:

        #     if self.social_init_type == 'res':
        #         target_pos = self.res_pos
        #     elif self.social_init_type == 'ag':
        #         target_pos = self.agents.sprites()[0].position

        #     spawn_x_min = max(x_min,target_pos[0] - self.social_init_range)
        #     spawn_x_max = min(x_max,target_pos[0] + self.social_init_range)
        #     spawn_y_min = max(x_min,target_pos[1] - self.social_init_range)
        #     spawn_y_max = min(x_max,target_pos[1] + self.social_init_range)
        #     # print(spawn_x_min, spawn_x_max, spawn_y_min, spawn_y_max)
            
        #     for i in range(1, self.N):

        #         colliding_resources = [0]
        #         colliding_agents = [0]

        #         retries = 0
        #         while len(colliding_resources) > 0 or len(colliding_agents) > 0 or dist_to_target > self.social_init_range:

        #             x = np.random.randint(spawn_x_min, spawn_x_max)
        #             y = np.random.randint(spawn_y_min, spawn_y_max)                
        #             orient = np.random.uniform(0, 2 * np.pi)
                    

        #             # print(x,y,i)

        #             agent = Agent(
        #                     id=i,
        #                     position=(x, y),
        #                     orientation=orient,
        #                     max_vel=self.max_vel,
        #                     FOV=self.agent_fov,
        #                     vision_range=self.vision_range,
        #                     num_class_elements=self.num_class_elements,
        #                     vis_field_res=self.vis_field_res,
        #                     consumption=self.agent_consumption,
        #                     model=None,
        #                     boundary_endpts=self.boundary_endpts,
        #                     window_pad=self.window_pad,
        #                     radius=self.agent_radii,
        #                     color=colors.CYAN,
        #                     vis_transform=self.vis_transform,
        #                     percep_angle_noise_std=self.percep_angle_noise_std,
        #                     sim_type=self.sim_type
        #                 )
                    
        #             colliding_resources = pygame.sprite.spritecollide(agent, self.resources, False, pygame.sprite.collide_circle)
        #             if self.agent_collide:
        #                 colliding_agents = pygame.sprite.spritecollide(agent, self.agents, False, supcalc.within_group_collision)
        #             else:
        #                 colliding_agents = []
        #             dist_to_target = supcalc.distance(agent.position, target_pos)

        #             retries += 1
        #             if retries > 1000: print(f'Retries > 1000')
        #         self.agents.add(agent)
        #         if 'ghost' in self.sim_type:
        #             self.viewable_agents.add(agent)

        # # ghost agent
        # if 'ghost' in self.sim_type:

        #     if 'offset' in self.sim_type:
        #         x,y = 600,600
        #     else:
        #         x,y = self.res_pos

        #     if 'exploiters' in self.sim_type or 'explorers' in self.sim_type: # multiple ghosts
        #         spacings = [] # 5*72 degree spacings, 1.5*agent_radius away from center as minimum needed to avoid collision
        #         for i in range(5):
        #             angle = i * 2*np.pi/5
        #             offset_x = x + 1.5*self.agent_radii*np.cos(angle)
        #             offset_y = y + 1.5*self.agent_radii*np.sin(angle)
        #             spacings.append((offset_x, offset_y))
        #         for i,j in spacings:
        #             agent = Agent(
        #                     id=self.N+1,
        #                     position=(i,j),
        #                     orientation=0,
        #                     max_vel=self.max_vel,
        #                     FOV=self.agent_fov,
        #                     vision_range=self.vision_range,
        #                     num_class_elements=self.num_class_elements,
        #                     vis_field_res=self.vis_field_res,
        #                     consumption=self.agent_consumption,
        #                     model=self.model,
        #                     boundary_endpts=self.boundary_endpts,
        #                     window_pad=self.window_pad,
        #                     radius=self.agent_radii,
        #                     color=colors.BLUE,
        #                     vis_transform=self.vis_transform,
        #                     percep_angle_noise_std=self.percep_angle_noise_std,
        #                     sim_type=self.sim_type,
        #                 )
        #             if 'exploiter' in self.sim_type:
        #                 agent.mode = 'exploit'
        #             elif 'explorer' in self.sim_type:
        #                 agent.mode = 'explore'
        #             self.ghost.add(agent)
        #             self.viewable_agents.add(agent)

        #     else: # single ghost
        #         agent = Agent(
        #                 id=self.N+1,
        #                 position=(x, y),
        #                 orientation=0,
        #                 max_vel=self.max_vel,
        #                 FOV=self.agent_fov,
        #                 vision_range=self.vision_range,
        #                 num_class_elements=self.num_class_elements,
        #                 vis_field_res=self.vis_field_res,
        #                 consumption=self.agent_consumption,
        #                 model=self.model,
        #                 boundary_endpts=self.boundary_endpts,
        #                 window_pad=self.window_pad,
        #                 radius=self.agent_radii,
        #                 color=colors.BLUE,
        #                 vis_transform=self.vis_transform,
        #                 percep_angle_noise_std=self.percep_angle_noise_std,
        #                 sim_type=self.sim_type,
        #             )
        #         if 'exploiter' in self.sim_type:
        #             agent.mode = 'exploit'
        #         elif 'explorer' in self.sim_type:
        #             agent.mode = 'explore'
        #         self.ghost.add(agent)
        #         self.viewable_agents.add(agent)

    # @timer
    def save_data_agent(self):
        agent = self.agents.sprites()[0] # only track 1st agent
        self.data_agent[self.t,:2] = agent.pt_eye
        self.data_agent[self.t,2] = agent.orientation

        if 'agent_explore' in agent.vis_field:
            self.data_agent[self.t,3] = 1
        else:
            self.data_agent[self.t,3] = 0

# ### -------------------------- RESOURCE FUNCTIONS -------------------------- ###

#     # @timer
#     def create_resources(self):

#         # creates single resource patch
#         id = 0
#         units = np.random.randint(self.min_res_units, self.max_res_units)
#         quality = np.random.uniform(self.min_res_quality, self.max_res_quality)

#         resource = Resource(id, self.res_radius, self.res_pos, units, quality)
#         self.resources.add(resource)

#         x,y = resource.position
#         pos_x = x
#         pos_y = self.y_max - y
#         self.data_res.append([pos_x, pos_y, self.res_radius])

#     def consume(self, agent):

#         # Call resource agent is on
#         resource = agent.res_to_be_consumed

#         # Increment remaining resource quantity
#         depl_units, destroy_res = resource.deplete(agent.consumption)

#         # Update agent info
#         if depl_units > 0:
#             agent.collected_r += depl_units
#             agent.mode = 'exploit'
#         else:
#             agent.mode = 'explore'

#         # Kill + regenerate patch when fully depleted
#         if destroy_res:
#             resource.kill()
#             if self.regenerate_resources:
#                 # self.add_new_resource_patch_random()
#                 self.add_new_resource_patch_stationary_single()

# ### -------------------------- COLLISION FUNCTIONS -------------------------- ###

#     # @timer
#     def collide_agent_res(self):

#         # Create dict of every agent that has collided : [colliding resources]
#         collision_group_ar = pygame.sprite.groupcollide(self.agents, self.resources, False, False, pygame.sprite.collide_circle)

#         # Switch on all agents currently on a resource 
#         for agent, resource_list in collision_group_ar.items():
#             for resource in resource_list:
#                 # Flip bool variable if agent is within patch boundary
#                 if supcalc.distance(agent.position, resource.position) <= resource.radius:
#                     agent.mode = 'exploit'
#                     agent.on_res = 1
#                     agent.res_to_be_consumed = resource
#                     break


#     def collide_agent_resarea(self): # for trajs.py where there is no self.resources

#         for i in range(1, self.N): # other agents only

#             agent = self.agents.sprites()[i]
#             dist_to_res = supcalc.distance(agent.position, self.res_pos)

#             if dist_to_res <= self.res_radius:
#                 agent.mode = 'exploit'


#     def collide_agent_wall(self):
        
#         # Create dict of every agent that has collided : [colliding walls]
#         collision_group_aw = pygame.sprite.groupcollide(self.agents, self.walls, False, False)

#         # Change agent mode + note points of contact (carry out velocity-stopping check later in agent.move())
#         for agent, wall_list in collision_group_aw.items():

#             agent.mode = 'collide'

#             for wall in wall_list:

#                 clip = agent.rect.clip(wall.rect)
#                 if self.with_visualization: pygame.draw.rect(self.screen, pygame.Color('red'), clip)

#                 # print(f'agent {agent.rect.center, agent.position} collided with {wall.id} @ {clip.center}')

#                 agent.collided_points.append(np.array(clip.center) - self.window_pad)

#                 # hits = [edge for edge in ['bottom', 'top', 'left', 'right'] if getattr(clip, edge) == getattr(agent.rect, edge)]
#                 # text = self.font.render(f'Collision at {", ".join(hits)}', True, pygame.Color('black'))
#                 # self.screen.blit(text, (self.window_pad, int(self.window_pad/2)))

#     def collide_agent_agent(self):

#         # Create dict of every agent that has collided : [colliding agents]
#         collision_group_aa = pygame.sprite.groupcollide(self.agents, self.agents, False, False, supcalc.within_group_collision)
        
#         # Carry out agent-agent collisions + generate list of collided agents
#         for agent1, other_agents in collision_group_aa.items():

#             if not self.agent_patch_collide:
#                 if agent1.mode == 'exploit':
#                     continue

#             agent1.mode = 'collide'

#             for agentX in other_agents:

#                 if not self.agent_patch_collide:
#                     if agentX.mode == 'exploit':
#                         continue

#                 clip = agent1.rect.clip(agentX.rect)
#                 if self.with_visualization: pygame.draw.rect(self.screen, pygame.Color('red'), clip)

#                 # print(f'agent {agent.rect.center} collided with {wall.id} @ {clip.center}')

#                 agent1.collided_points.append(np.array(clip.center) - self.window_pad)

#         if 'offset' in self.sim_type: # ghost outside patch --> acknowledge collisions, same code as above
#             collision_group_aa = pygame.sprite.groupcollide(self.agents, self.ghost, False, False, supcalc.within_group_collision)
#             for agent1, other_agents in collision_group_aa.items():
#                 if not self.agent_patch_collide:
#                     if agent1.mode == 'exploit':
#                         continue
#                 agent1.mode = 'collide'
#                 for agentX in other_agents:
#                     if not self.agent_patch_collide:
#                         if agentX.mode == 'exploit':
#                             continue
#                     clip = agent1.rect.clip(agentX.rect)
#                     if self.with_visualization: pygame.draw.rect(self.screen, pygame.Color('red'), clip)
#                     agent1.collided_points.append(np.array(clip.center) - self.window_pad)


### -------------------------- HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def interact_with_event(self, events):
        """Carry out functionality according to user's interaction"""
        for event in events:
            # Exit if requested
            if event.type == pygame.QUIT:
                sys.exit()

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


##################################################################################
### -------------------------- MAIN SIMULATION LOOP -------------------------- ###
##################################################################################

    def start(self):

        ### ---- INITIALIZATION ---- ###

        start_time = time.time()
        # self.create_walls()
        # if self.agent_init is None:
        #     self.create_resources()
        self.create_agents()

        ### ---- START OF SIMULATION ---- ###

        while self.t < self.T:

            if not self.is_paused:

                # self.recorder.capture_frame(self.screen)
                
                ### ---- OBSERVATIONS ---- ###

                # # Refresh agent behavioral states
                # for agent in self.agents:
                    
                #     agent.collided_points = []
                #     if agent.mode == 'collide': agent.mode = 'explore'

                # # Evaluate sprite interactions + flip agent modes to 'collide'/'exploit' (latter takes precedence)
                # self.collide_agent_wall()
                # if self.agent_collide:
                #     self.collide_agent_agent()
                # if self.agent_init is None:
                #     self.collide_agent_res()
                # else:
                #     self.collide_agent_resarea()

                # Update visual projections
                for agent in self.agents:
                    if agent.id == 0: # only ANN agent
                        if 'ghost' in self.sim_type:
                            agent.visual_sensing(self.objs, self.viewable_agents)
                        else:
                            agent.visual_sensing(self.objs, self.agents)

                ### ---- VISUALIZATION ---- ###

                if self.with_visualization:
                    if 'ghost' in self.sim_type:
                        for agent in self.viewable_agents:
                            agent.draw_update() 
                    else:
                        for agent in self.agents:
                            agent.draw_update() 
                    for res in self.resources:
                        res.draw_update() 
                    self.draw_frame()
                else: # still have to update rect for collisions
                    for agent in self.agents:
                        agent.rect = agent.image.get_rect(center = agent.position + self.window_pad)

                ### ---- TRACKING ---- ### 

                self.save_data_agent()

                ### ---- MODEL + ACTIONS ---- ###

                for agent in self.agents:

                    # ANN agent
                    if agent.id == 0:

                        # # Food present --> terminate simulation
                        # if agent.mode == 'exploit':

                        #     # self.recorder.end_recording()
                        #     pygame.quit()
                        #     elapsed_time = round( (time.time() - start_time) , 2)
                        #     return self.t, 0, elapsed_time, self.data_agent

                        # else: # No food --> sense + move (via ANN)

                        agent.move(-self.spin_angle)

                    else: # RW/direct agents

                        # jiggle agent around static point
                        if self.agent_init == 0:
                            pass
                        else:
                            agent.position = np.array([1000 - self.agent_init + np.random.uniform(-25,25), 1000 - self.agent_init + np.random.uniform(-25,25)])
    
                    #     # Food present --> consume
                    #     if agent.mode == 'exploit':
                    #         # self.consume(agent) # does not do anything important + makes trajs.py tricky, thus taken out, change for patch depletion
                    #         pass

                    #     else: # No food --> move

                    #         if agent.id < self.N_rand + 1: # (via random walk)
                    #             agent.move((2*self.rot_diff)**.5 * np.random.uniform(-1,1))

                    #         else: # (via direct line)
                    #             if agent.collide_timer > 0:
                    #                 agent.collide_timer -= 1
                    #                 agent.move(0)
                    #             else:
                    #                 # turn toward res if facing away
                    #                 angle_to_food = np.arctan2(-(self.res_pos[1] - agent.position[1]),
                    #                                         self.res_pos[0] - agent.position[0])
                    #                 angle_diff = angle_to_food - agent.orientation
                    #                 angle_diff = (angle_diff - np.pi) % (2*np.pi) - np.pi
                    #                 if np.abs(angle_diff) > 0.01: # angular tolerance
                    #                     agent.move(angle_diff)
                    #                 else: # or move fwd
                    #                     agent.move(0)
                                

            ### ---- BACKGROUND PROCESSES ---- ###
        
                # Step sim time forward
                self.t += 1

                # Step clock time to calculate fps
                if self.with_visualization:
                    self.clock.tick(self.framerate)
                    # print(f"t={self.t} \t| FPS: {round(self.clock.get_fps(),1)}")

            # Carry out user interactions even when not paused
            if self.with_visualization:
                events = pygame.event.get() 
                self.interact_with_event(events)

        ### ---- END OF SIMULATION ---- ###

        # self.recorder.end_recording()
        pygame.quit()
        elapsed_time = round( (time.time() - start_time) , 2)
        dist_to_res = supcalc.distance(self.agents.sprites()[0].position, self.res_pos)
        return self.T, dist_to_res, elapsed_time, self.data_agent