import contextlib
with contextlib.redirect_stdout(None): # blocks pygame initialization messages
    import pygame

import numpy as np
import sys
# import os
# import uuid
from datetime import datetime
import time

from abm.agent import supcalc
from abm.agent.agent import Agent
from abm.environment.resource import Resource
from abm.contrib import colors
from abm.monitoring import tracking, plot_funcs
# from abm.monitoring import env_saver

class Simulation:
    def __init__(self, width=600, height=480, window_pad=30, 
                 N=1, T=1000, with_visualization=True, framerate=25, print_enabled=False, plot_trajectory=False, 
                 log_zarr_file=False, sim_save_name=None,
                 agent_radius=10, max_vel=5, collision_slowdown=0.5, vis_field_res=8, contact_field_res=4, collide_agents=True, 
                 vision_range=150, agent_fov=1.0, visual_exclusion=False, show_vision_range=False, agent_consumption=1, 
                 N_resrc=10, patch_radius=30, min_resrc_perpatch=200, max_resrc_perpatch=1000, 
                 min_resrc_quality=0.1, max_resrc_quality=1, regenerate_patches=True, 
                 NN=None, NN_weight_init=None, NN_input_other_size=3, NN_hidden_size=128, NN_output_size=1
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
        :param sim_save_name:
        :param agent_radius: radius of the agents
        :param max_vel:
        :param collision_slowdown:
        :param vis_field_res: projection field (visual + proximity) resolution in pixels
        :param contact_field_res:
        :param collide_agents: boolean switch agents can overlap if false
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
        :param NN_weight_init:
        :param NN_input_other_size:
        :param NN_hidden_size:
        :param NN_output_size:
        """
        # Arena parameters
        self.WIDTH = width
        self.HEIGHT = height
        self.window_pad = window_pad
        self.x_min, self.x_max = window_pad, window_pad + width # [30, 430]
        self.y_min, self.y_max = window_pad, window_pad + height # [30, 430]
        self.boundary_info = (self.x_min, self.x_max, self.y_min, self.y_max)

        # Simulation parameters
        self.N = N
        self.T = T
        self.t = 0
        self.with_visualization = with_visualization
        if self.with_visualization:
            self.framerate_orig = framerate
        else:
            # this is more than what is possible withy pygame so it will use the maximal framerate
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
        self.crash = False
        self.sim_save_name = sim_save_name

        # Agent parameters
        self.agent_radii = agent_radius
        self.max_vel = max_vel
        self.collision_slowdown = collision_slowdown
        self.vis_field_res = vis_field_res
        self.contact_field_res = contact_field_res
        self.collide_agents = collide_agents
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
        self.res_id_counter = 0

        # Neural Network parameters
        self.NN = NN
        self.NN_weight_init = NN_weight_init

        if N == 1:  num_class_elements = 4 # single-agent --> perception of 4 walls
        else:       num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes

        self.vis_size = vis_field_res * num_class_elements
        self.contact_size = contact_field_res * num_class_elements
        self.other_size = NN_input_other_size # on_resrc + velocity + orientation
        
        self.NN_input_size = self.vis_size + self.contact_size + self.other_size
        self.NN_hidden_size = NN_hidden_size
        self.NN_output_size = NN_output_size # dvel + dtheta
        
        if print_enabled: 
            print(f"NN inputs = {self.vis_size} (vis_size) + {self.contact_size} (contact_size)",end="")
            print(f" + {self.other_size} (velocity + orientation) = {self.NN_input_size}")
            print(f"Agent NN architecture = {(self.NN_input_size, NN_hidden_size, self.NN_output_size)}")

        # Initializing pygame
        if self.with_visualization:
            pygame.init()
            self.screen = pygame.display.set_mode([self.x_min + self.x_max, self.y_min + self.y_max])
        else:
            pygame.display.init()
            pygame.display.set_mode([1,1])

        # pygame related class attributes
        self.agents = pygame.sprite.Group()
        self.resources = pygame.sprite.Group()
        self.clock = pygame.time.Clock() # todo: look into this more in detail so we can control dt

### -------------------------- DRAWING FUNCTIONS -------------------------- ###

    def draw_walls(self):
        """Drawing walls on the arena according to initialization"""
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.x_min, self.y_min],
                         [self.x_min, self.y_max])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.x_min, self.y_min],
                         [self.x_max, self.y_min])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.x_max, self.y_min],
                         [self.x_max, self.y_max])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.x_min, self.y_max],
                         [self.x_max, self.y_max])

    def draw_framerate(self):
        """Showing framerate, sim time and pause status on simulation windows"""
        tab_size = self.window_pad
        line_height = int(self.window_pad / 2)
        font = pygame.font.Font(None, line_height)
        status = [
            f"FPS: {self.framerate}, t = {self.t}/{self.T}",
        ]
        if self.is_paused:
            status.append("-Paused-")
        for i, stat_i in enumerate(status):
            text = font.render(stat_i, True, colors.BLACK)
            self.screen.blit(text, (tab_size, i * line_height))

    def draw_agent_stats(self, font_size=15, spacing=0):
        """Showing agent information when paused"""
        font = pygame.font.Font(None, font_size)
        for agent in self.agents:
            if agent.is_moved_with_cursor or agent.show_stats:
                status = [
                    f"ID: {agent.id}",
                    f"res.: {agent.collected_r:.2f}",
                    f"ori.: {agent.orientation:.2f}",
                    f"w: {agent.w:.2f}"
                ]
                for i, stat_i in enumerate(status):
                    text = font.render(stat_i, True, colors.BLACK)
                    self.screen.blit(text, (agent.position[0] + 2 * agent.radius,
                                            agent.position[1] + 2 * agent.radius + i * (font_size + spacing)))

    def draw_visual_fields(self):
        """Visualizing range of vision as opaque circles around the agents""" 
        for agent in self.agents:
            # Show visual range as circle if non-limiting FOV
            if self.agent_fov == 1:
                pygame.draw.circle(self.screen, colors.GREY, agent.pt_eye, agent.vision_range, width=1)
            else: # self.agent_fov < 1 --> show limits of FOV as radial lines with length of visual range
                start_pos = agent.pt_eye
                angles = (agent.orientation + agent.phis[0], 
                          agent.orientation + agent.phis[-1])
                for angle in angles: ### draws lines that don't quite meet borders
                    end_pos = (start_pos[0] + np.cos(angle) * agent.vision_range,
                               start_pos[1] - np.sin(angle) * agent.vision_range)
                    pygame.draw.line(self.screen, colors.GREY, start_pos, end_pos, 1)

            # Show what it is seeing as discretized circles, color reflects identity
            for phi, vis_name in zip(agent.phis, agent.vis_field):

                if vis_name == 'wall_north': # --> red
                    pygame.draw.circle(
                        self.screen, colors.TOMATO, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * agent.vision_range,
                         start_pos[1] - np.sin(agent.orientation - phi) * agent.vision_range),
                        radius=3)
                elif vis_name == 'wall_south': # --> green
                    pygame.draw.circle(
                        self.screen, colors.LIME, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * agent.vision_range,
                         start_pos[1] - np.sin(agent.orientation - phi) * agent.vision_range),
                        radius=3)
                elif vis_name == 'wall_east': # --> blue
                    pygame.draw.circle(
                        self.screen, colors.CORN, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * agent.vision_range,
                         start_pos[1] - np.sin(agent.orientation - phi) * agent.vision_range),
                        radius=3)
                elif vis_name == 'wall_west': # --> yellow
                    pygame.draw.circle(
                        self.screen, colors.GOLD, 
                        (start_pos[0] + np.cos(agent.orientation - phi) * agent.vision_range,
                         start_pos[1] - np.sin(agent.orientation - phi) * agent.vision_range),
                        radius=3)
                # elif vis_name == 'exploit': ax.scatter(orient-phi, 600, s=20, c='k')
                # else: # non-exploiting 
                #     ax.scatter(orient-phi, 600, s=20, c='y')
    
    def draw_frame(self):
        """Drawing environment, agents and every other visualization in each timestep"""
        pygame.display.flip()
        self.screen.fill(colors.BACKGROUND)
        self.resources.draw(self.screen)
        self.agents.draw(self.screen)
        self.draw_walls()
        self.draw_framerate()
        self.draw_agent_stats()

        
        
        # vision range + projection field
        if self.show_vision_range and self.x_max > self.vision_range: 
            self.draw_visual_fields()

### -------------------------- AGENT FUNCTIONS -------------------------- ###

    def create_agents(self):
        """Creating agents according to how the simulation class was initialized"""
        for i in range(self.N):
            # allowing agents to overlap arena borders (maximum overlap is radius of patch)
            x = np.random.randint(self.x_min - self.agent_radii, self.x_max - self.agent_radii)
            y = np.random.randint(self.y_min - self.agent_radii, self.y_max - self.agent_radii)
            orient = np.random.uniform(0, 2 * np.pi) # randomly orients according to 0,pi/2 : right,up
            self.agents.add(
                Agent(
                    id=i,
                    position=(x, y),
                    orientation=orient,
                    max_vel=self.max_vel,
                    collision_slowdown=self.collision_slowdown,
                    vis_field_res=self.vis_field_res,
                    FOV=self.agent_fov,
                    vision_range=self.vision_range,
                    visual_exclusion=self.visual_exclusion,
                    contact_field_res=self.contact_field_res,
                    consumption=self.agent_consumption,
                    vis_size=self.vis_size,
                    contact_size=self.contact_size,
                    NN_input_size=self.NN_input_size,
                    NN_hidden_size=self.NN_hidden_size,
                    NN_output_size=self.NN_output_size,
                    NN=self.NN,
                    NN_weight_init=self.NN_weight_init,
                    boundary_info=self.boundary_info,
                    radius=self.agent_radii,
                    color=colors.BLUE,
                )
            )

    def save_data_agent(self):

        for agent in self.agents:
        
            # Gather tracked variables + enter into save array at current timestep
            x, y = agent.pt_center
            pos_x = x - self.window_pad
            pos_y = self.y_max - y

            if agent.mode == 'explore': mode_num = 0
            elif agent.mode == 'exploit': mode_num = 1
            elif agent.mode == 'collide': mode_num = 2
            else: raise Exception('Agent Mode not tracked')

            self.data_agent[agent.id, self.t, :] = np.array((pos_x, pos_y, mode_num, agent.collected_r))

### -------------------------- RESOURCE FUNCTIONS -------------------------- ###

    def create_resources(self):
        """Creating resource patches according to how the simulation class was initialized"""
        for i in range(self.N_resrc):
            self.add_new_resource_patch()

    def add_new_resource_patch(self):
        """Adding a new resource patch to the resources sprite group. The position of the new resource is proved with
        prove_resource method so that the distribution and overlap is following some predefined rules"""
        # takes current id, increments counter for next patch
        id = self.res_id_counter
        self.res_id_counter += 1

        radius = self.resrc_radius

        max_retries = 100
        retries = 0
        colliding_resources = [0]
        colliding_agents = [0]

        while len(colliding_resources) > 0 or len(colliding_agents) > 0:
            if retries > max_retries:
                raise Exception("Reached timeout while trying to create resources without overlap!")
            
            x = np.random.randint(self.x_min, self.x_max - 2*radius)
            y = np.random.randint(self.y_min, self.y_max - 2*radius)

            units = np.random.randint(self.min_resrc_units, self.max_resrc_units)
            quality = np.random.uniform(self.min_resrc_quality, self.max_resrc_quality)
            resource = Resource(id, radius, (x, y), units, quality)

            # check for overlap with other resources + agents
            colliding_resources = pygame.sprite.spritecollide(resource, self.resources, False, pygame.sprite.collide_circle)
            colliding_agents = pygame.sprite.spritecollide(resource, self.agents, False, pygame.sprite.collide_circle)
            retries += 1

        # adds new resources when overlap is no longer detected
        self.resources.add(resource)

        if not self.log_zarr_file: # save in sim instance

            # convert positional coordinates
            x,y = resource.pt_center
            pos_x = x - self.window_pad
            pos_y = self.y_max - y

            self.data_res.append([pos_x, pos_y, radius])

    def consume(self, agent):
        """
        Carry out agent-resource interactions (depletion, destroying, notifying)
        """
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
                # self.add_new_resource_patch(force_id=resource.id)
                self.add_new_resource_patch()

### -------------------------- MULTIAGENT INTERACTION FUNCTIONS -------------------------- ###

    def collide_agent_res(self):

        # Create dict of every agent that has collided : [colliding resources]
        collision_group_ar = pygame.sprite.groupcollide(self.agents, self.resources, False, False,
                                                        pygame.sprite.collide_circle)

        # Switch on all agents currently on a resource 
        for agent, resource_list in collision_group_ar.items():
            for resource in resource_list:
                # Flip bool variable if agent is within patch boundary
                if supcalc.distance(agent.pt_center, resource.pt_center) <= resource.radius:
                    agent.on_resrc = 1
                    agent.res_to_be_consumed = resource
                    break

    # def collide_agent_agent(self):

        # # Create dict of every agent that has collided : [colliding agents]
        # collision_group_aa = pygame.sprite.groupcollide(self.agents, self.agents, False, False,
        #     itra.within_group_collision)
        
        # # Carry out agent-agent collisions + generate list of non-exploiting collided agents
        # for agent1, other_agents in collision_group_aa.items():
        #     collided_agents_instance = self.agent_agent_collision_proximity(agent1, other_agents)
        #     collided_agents.append(collided_agents_instance)
        # collided_agents = [agent for sublist in collided_agents for agent in sublist]
        
        # # Turn off collision mode when over
        # for agent in self.agents:
        #     if agent not in collided_agents and agent.get_mode() == "collide":
        #         agent.set_mode("explore")

### -------------------------- HUMAN INTERACTION FUNCTIONS -------------------------- ###

    def interact_with_event(self, events):
        """Carry out functionality according to user's interaction"""
        for event in events:
            # Exit if requested
            if event.type == pygame.QUIT:
                sys.exit()

            # Change orientation with mouse wheel
            if event.type == pygame.MOUSEWHEEL:
                if event.y == -1:
                    event.y = 0
                for ag in self.agents:
                    ag.move_with_mouse(pygame.mouse.get_pos(), event.y, 1 - event.y)

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
                if self.framerate > 35:
                    self.framerate = 35
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.framerate = self.framerate_orig

            # Continuous mouse events (move with cursor)
            if pygame.mouse.get_pressed()[0]:
                try:
                    for ag in self.agents:
                        ag.move_with_mouse(event.pos, 0, 0)
                    for res in self.resources:
                        res.update_clicked_status(event.pos)
                except AttributeError:
                    for ag in self.agents:
                        ag.move_with_mouse(pygame.mouse.get_pos(), 0, 0)
            else:
                for ag in self.agents:
                    ag.is_moved_with_cursor = False
                    ag.draw_update()
                for res in self.resources:
                    res.is_clicked = False
                    res.draw_update()


##################################################################################
### -------------------------- MAIN SIMULATION LOOP -------------------------- ###
##################################################################################

    def start(self):

        ### ---- INITIALIZATION ---- ###

        start_time = datetime.now()
        self.create_agents()
        self.create_resources()

        ### ---- START OF SIMULATION ---- ###

        while self.t < self.T:

            if not self.is_paused:

                ### ---- OBSERVATIONS ---- ###

                for agent in self.agents:

                    # Update agent color/position/on_resrc parameter
                    agent.draw_update() 
                    agent.on_resrc = 0

                    # Update visual projections (vis_field)
                    crash = agent.visual_sensing()
                    
                    if crash: # position = nan : RNN weight explosion due to spinning
                        pygame.quit()
                        tracking.clean_global_dicts() # clean global data structures
                        self.crash = True
                        return 0,0,self.crash

                    # Update collisions with walls (contact_field)
                    agent.wall_contact_sensing() 

                # Update collisions with resources (on_resrc)
                self.collide_agent_res()
                    
                # # Update collisions amongst agents (contact_field)
                # if self.collide_agents:
                #     self.collide_agent_agent()

                ### ---- VISUALIZATION ---- ###

                if self.with_visualization:
                    self.draw_frame()

                ### ---- TRACKING ---- ### 

                if self.log_zarr_file:
                    tracking.save_agent_data_RAM(self)
                    tracking.save_resource_data_RAM(self)
                else:
                    self.save_data_agent()

                ### ---- MODEL + ACTIONS ---- ###

                for agent in self.agents:

                    # Pass observables through NN to calculate actions (dvel + dtheta) & advance agent's hidden state
                    NN_input = agent.assemble_NN_inputs()
                    NN_output, agent.hidden = agent.NN.forward(NN_input, agent.hidden)

                    # Food present --> consume + set mode to exploit (if food is still available)
                    if agent.on_resrc == 1:
                    # if agent.on_resrc == 1 and agent.velocity == 0: 
                        self.consume(agent) 

                    # No food --> move via decided actions + set mode to collide if collided
                    else: agent.move(NN_output) 

            ### ---- BACKGROUND PROCESSES ---- ###
        
                # Step simulation time forward
                self.t += 1

                # Step clock time to calculate fps
                self.clock.tick(self.framerate)
                if self.print_enabled and (self.t % 500 == 0):
                    print(f"t={self.t} \t| FPS: {self.clock.get_fps()}")

            # Carry out user interactions even when not paused
            if self.with_visualization:
                events = pygame.event.get() 
                self.interact_with_event(events)

        ### ---- END OF SIMULATION ---- ###

        pygame.quit()

        # compute simulation time in seconds
        self.elapsed_time = (datetime.now() - start_time).total_seconds()
        if self.print_enabled:
            print(f"Elapsed_time: {self.elapsed_time}")

        # conclude agent/resource tracking
        if self.log_zarr_file:
            # convert tracking agent/resource dicts to N-dimensional zarr arrays + save to offline file
            ag_zarr, res_zarr = tracking.save_zarr_file(self.T, self.sim_save_name)

            # assign plot data as zarr arrays
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
            plot_funcs.plot_map(plot_data, self.WIDTH, self.HEIGHT)

        return self.fitnesses, self.elapsed_time, False