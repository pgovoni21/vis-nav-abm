import contextlib
with contextlib.redirect_stdout(None): # blocks pygame initialization messages
    import pygame

import numpy as np
import sys
import os
import uuid
from datetime import datetime
import matplotlib.pyplot as plt

from abm.agent import supcalc
from abm.agent.agent import Agent
from abm.environment.resource import Resource
from abm.contrib import colors, ifdb_params, evolution
from abm.simulation import interactions as itra
from abm.monitoring import ifdb
from abm.monitoring import env_saver

class Simulation:
    def __init__(self, N, T, vis_field_res=800, width=600, height=480,
                 framerate=25, window_pad=30, with_visualization=True, show_vis_field=False,
                 show_vis_field_return=False, agent_radius=10, max_vel=5, collision_slowdown=0.5,
                 N_resrc=10, min_resrc_perpatch=200, max_resrc_perpatch=1000, min_resrc_quality=0.1, max_resrc_quality=1,
                 patch_radius=30, regenerate_patches=True, agent_consumption=1, 
                 vision_range=150, agent_fov=1.0, visual_exclusion=False, show_vision_range=False,
                 use_ifdb_logging=False, use_ram_logging=False, save_csv_files=False, parallel=False, use_zarr=True, write_batch_size=100,
                 collide_agents=True, contact_field_res=4, NN=None, NN_weight_init=None, NN_hidden_size=128, 
                 print_enabled=False, plot_trajectory=False):
        """
        Initializing the main simulation instance
        :param N: number of agents
        :param T: simulation time
        :param vis_field_res: projection field (visual + proximity) resolution in pixels
        :param width: real width of environment (not window size)
        :param height: real height of environment (not window size)
        :param framerate: framerate of simulation
        :param window_pad: padding of the environment in simulation window in pixels
        :param with_visualization: turns visualization on or off. For large batch autmatic simulation should be off so
            that we can use a higher/maximal framerate.
        :param show_vis_field: (Bool) turn on visualization for visual field of agents
        :param show_vis_field_return: (Bool) sow visual fields when return/enter is pressed
        :param agent_radius: radius of the agents
        :param N_resrc: number of resource patches in the environment
        :param min_resrc_perpatch: minimum resource unit per patch
        :param max_resrc_perpatch: maximum resource units per patch
        :param min_resrc_quality: minimum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        : param max_resrc_quality: maximum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        :param patch_radius: radius of resrcaurce patches
        :param regenerate_patches: bool to decide if patches shall be regenerated after depletion
        :param agent_consumption: agent consumption (exploitation speed) in res. units / time units
        :param vision_range: range (in px) of agents' vision
        :param agent_fov (float): the field of view of the agent as percentage. e.g. if 0.5, the the field of view is
                                between -pi/2 and pi/2
        :param visual_exclusion: when true agents can visually exclude socially relevant visual cues from other agents'
                                projection field
        :param show_vision_range: bool to switch visualization of visual range for agents. If true the limit of far
                                and near field visual field will be drawn around the agents
        :param use_ifdb_logging: Switch to turn IFDB save on or off
        :param use_ram_logging: log data into memory (RAM) only use this if ifdb is problematic and you have enough
            resources
        :param save_csv_files: Save all recorded IFDB data as csv file. Only works if IFDB looging was turned on
        :param parallel: if True we request to run the simulation parallely with other simulation instances and hence
            the influxDB saving will be handled accordingly.
        :param use_zarr: using zarr compressed data format to save single run data
        :param allow_border_patch_overlap: boolean switch to allow resource patches to overlap arena border
        :param collide_agents: boolean switch agents can overlap if false
        :param print_enabled:
        :param plot_trajectory:
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

        # Visualization parameters
        self.show_vis_field = show_vis_field
        self.show_vis_field_return = show_vis_field_return
        self.show_vision_range = show_vision_range

        # Agent parameters
        self.agent_radii = agent_radius
        self.max_vel = max_vel
        self.collision_slowdown = collision_slowdown
        self.vis_field_res = vis_field_res
        self.contact_field_res = contact_field_res
        self.agent_consumption = agent_consumption
        self.vision_range = vision_range
        self.agent_fov = agent_fov
        self.visual_exclusion = visual_exclusion
        self.collide_agents = collide_agents

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

        # Neural Network parameters
        self.NN = NN
        self.NN_weight_init = NN_weight_init

        if N == 1:
            num_class_elements = 4 # single-agent --> perception of 4 walls
        else:
            num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes

        self.vis_size = vis_field_res * num_class_elements
        self.contact_size = contact_field_res * num_class_elements
        self.other_size = 2 # velocity + orientation
        # self.other_size = 3 # on_resrc + velocity + orientation
        
        self.NN_input_size = self.vis_size + self.contact_size + self.other_size
        self.NN_hidden_size = NN_hidden_size # input via env variable
        self.NN_output_size = 2 # dvel + dtheta
        
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

        # # Monitoring
        # self.use_zarr = use_zarr
        # self.write_batch_size = write_batch_size
        # self.parallel = parallel
        # if self.parallel:
        #     self.ifdb_hash = uuid.uuid4().hex
        # else:
        #     self.ifdb_hash = ""
        # self.save_in_ifd = use_ifdb_logging
        # self.save_in_ram = use_ram_logging
        # self.save_csv_files = save_csv_files
        # if self.save_in_ram:
        #     self.save_in_ifd = False
        #     if print_enabled: print("Turned off IFDB logging as RAM logging was explicitly requested!!!")

        # if self.save_in_ifd:
        #     self.ifdb_client = ifdb.create_ifclient()
        #     if not self.parallel:
        #         self.ifdb_client.drop_database(ifdb_params.INFLUX_DB_NAME)
        #     self.ifdb_client.create_database(ifdb_params.INFLUX_DB_NAME)
        #     # ifdb.save_simulation_params(self.ifdb_client, self, exp_hash=self.ifdb_hash)
        # else:
        #     self.ifdb_client = None

        # # by default we parametrize with the .env file in root folder
        # EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
        # root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        # self.env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")

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
    
    def draw_frame(self, stats, stats_pos):
        """Drawing environment, agents and every other visualization in each timestep"""
        self.screen.fill(colors.BACKGROUND)
        self.resources.draw(self.screen)
        self.agents.draw(self.screen)
        self.draw_walls()
        self.draw_framerate()
        self.draw_agent_stats()
        
        # vision range + projection field
        if self.show_vision_range and self.x_max > self.vision_range: 
            self.draw_visual_fields()

        # # visual field graphs of each agent
        # if self.show_vis_field:
        #     self.update_vis_field_graph(stats, stats_pos)

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
            
    def create_vis_field_graph(self):
        """Creating visualization graph for visual fields of the agents"""
        stats = pygame.Surface((self.WIDTH, 50 * self.N))
        stats.fill(colors.GREY)
        stats.set_alpha(150)
        stats_pos = (int(self.window_pad), int(self.window_pad))
        return stats, stats_pos

    # def update_vis_field_graph(self, stats, stats_pos):
    #     """Showing visual fields of the agents on a specific graph"""
    #     stats_width = stats.get_width()
    #     # Updating our graphs to show visual field
    #     stats_graph = pygame.PixelArray(stats)
    #     stats_graph[:, :] = pygame.Color(*colors.WHITE)
    #     for k in range(self.N):
    #         show_base = k * 50
    #         show_min = (k * 50) + 23
    #         show_max = (k * 50) + 25

    #         for j in range(self.agents.sprites()[k].vis_field_res):
    #             curr_idx = int(j * (stats_width / self.vis_field_res))
    #             if self.agents.sprites()[k].soc_vis_field[j] != 0:
    #                 stats_graph[curr_idx, show_min:show_max] = pygame.Color(*colors.GREEN)
    #             # elif self.agents.sprites()[k].soc_v_field[j] == -1:
    #             #     stats_graph[j, show_min:show_max] = pygame.Color(*colors.RED)
    #             else:
    #                 stats_graph[curr_idx, show_base] = pygame.Color(*colors.GREEN)

    #     del stats_graph
    #     stats.unlock()

    #     # Drawing
    #     self.screen.blit(stats, stats_pos)
    #     for agi, ag in enumerate(self.agents):
    #         line_height = 15
    #         font = pygame.font.Font(None, line_height)
    #         status = f"agent {ag.id}"
    #         text = font.render(status, True, colors.BLACK)
    #         self.screen.blit(text, (int(self.window_pad) / 2, self.window_pad + agi * 50))

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
                    NN=self.NN,
                    NN_weight_init=self.NN_weight_init,
                    vis_size=self.vis_size,
                    contact_size=self.contact_size,
                    NN_input_size=self.NN_input_size,
                    NN_hidden_size=self.NN_hidden_size,
                    NN_output_size=self.NN_output_size,
                    boundary_info=self.boundary_info,
                    radius=self.agent_radii,
                    color=colors.BLUE,
                    print_enabled=self.print_enabled
                )
            )

### -------------------------- RESOURCE FUNCTIONS -------------------------- ###

    def create_resources(self):
        """Creating resource patches according to how the simulation class was initialized"""
        if self.plot_trajectory: 
            self.resrc_pos = []
        for i in range(self.N_resrc):
            self.add_new_resource_patch()

    def add_new_resource_patch(self, force_id=None):
        """Adding a new resource patch to the resources sprite group. The position of the new resource is proved with
        prove_resource method so that the distribution and overlap is following some predefined rules"""
        if force_id is None: # find new id
            if len(self.resources) > 0:
                id = max([resrc.id for resrc in self.resources]) + 1
            else:
                id = 0
        else:
            id = force_id

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
        
        if self.plot_trajectory: 
            self.resrc_pos.append(( x + radius, self.y_max - (y + radius) + self.y_min ))

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
                self.add_new_resource_patch(force_id=resource.id)

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

    def decide_on_vis_field_visibility(self, turned_on_vfield):
        """Deciding if the visual field needs to be shown or not"""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            show_vis_fields_on_return = self.show_vis_field_return
            if not self.show_vis_field and show_vis_fields_on_return:
                self.show_vis_field = 1
                turned_on_vfield = 1
        else:
            if self.show_vis_field and turned_on_vfield:
                turned_on_vfield = 0
                self.show_vis_field = 0
        return turned_on_vfield


    def plot_map(self, traj_explore, traj_exploit, traj_collide, w, h):

        fig, axes = plt.subplots() 

        # rescale plotting area to square
        l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
        fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

        # simulation boundaries via lines
        axes.plot( [self.x_min, self.x_max], [self.y_min, self.y_min], color='k')
        axes.plot( [self.x_min, self.x_max], [self.y_max, self.y_max], color='k')
        axes.plot( [self.x_min, self.x_min], [self.y_min, self.y_max], color='k')
        axes.plot( [self.x_max, self.x_max], [self.y_min, self.y_max], color='k')

        # resource patches via circles
        for x,y in self.resrc_pos: 
            axes.add_patch( plt.Circle((x,y),self.resrc_radius, color='lightgray', zorder=0) )

        # convert agent data to arrays to exploit vectorization
        traj_explore = np.array(traj_explore)
        traj_exploit = np.array(traj_exploit)
        traj_collide = np.array(traj_collide)

        # agent directional trajectory via arrows 
        self.arrows(axes, traj_explore)

        # agent positional trajectory + mode via points
        axes.plot(traj_explore[:,0], traj_explore[:,1], 'o', color='royalblue', ms=.5, zorder=2)
        axes.plot(traj_exploit[:,0], traj_exploit[:,1], 'o', color='green', ms=5, zorder=2)
        axes.plot(traj_collide[:,0], traj_collide[:,1], 'o', color='red', ms=5, zorder=2)

        # agent start/end points
        axes.plot(traj_explore[0,0], traj_explore[0,1], 'wo', ms=10, markeredgecolor='k', zorder=4)
        axes.plot(traj_explore[-1,0], traj_explore[-1,1], 'ko', ms=10, zorder=4)

        axes.set_xlim(0,self.x_max + self.window_pad)
        axes.set_ylim(0,self.y_max + self.window_pad)

        plt.show()

        # for live plotting during an evol run - pull details to EA class for continuous live plotting
        # implement this --> https://stackoverflow.com/a/49594258
        # plt.clf
        # plt.draw()
        # plt.pause(0.001)

    def arrows(self, axes, traj_data, ahl=6, ahw=3):
        # from here: https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib

        x = traj_data[:,0]
        y = traj_data[:,1]

        # r is the distance spanned between pairs of points
        r = [0]
        for i in range(1,len(x)):
            dx = x[i]-x[i-1]
            dy = y[i]-y[i-1]
            r.append(np.sqrt(dx*dx+dy*dy))
        r = np.array(r)

        # set arrow spacing
        num_arrows = int(self.T / 15)
        aspace = r.sum() / (num_arrows + 1)
        
        # set inital arrow position at first space
        arrowPos = 0
        
        # rtot is a cumulative sum of r, it's used to save time
        rtot = []
        for i in range(len(r)):
            rtot.append(r[0:i].sum())
        rtot.append(r.sum())

        arrowData = [] # will hold tuples of x,y,theta for each arrow

        ndrawn = 0
        rcount = 1 
        while arrowPos < r.sum() and ndrawn < num_arrows:
            x1, x2 = x[rcount-1], x[rcount]
            y1, y2 = y[rcount-1], y[rcount]
            da = arrowPos - rtot[rcount]
            theta = np.arctan2((x2-x1),(y2-y1))
            ax = np.sin(theta)*da + x1
            ay = np.cos(theta)*da + y1
            arrowData.append((ax,ay,theta))
            ndrawn += 1
            arrowPos += aspace
            while arrowPos > rtot[rcount+1]: 
                rcount += 1
                if arrowPos > rtot[-1]:
                    break

        for ax,ay,theta in arrowData:
            # use aspace as a guide for size and length of things
            # scaling factors were chosen by experimenting a bit

            dx0 = np.sin(theta)*ahl/2. + ax
            dy0 = np.cos(theta)*ahl/2. + ay
            dx1 = -1.*np.sin(theta)*ahl/2. + ax
            dy1 = -1.*np.cos(theta)*ahl/2. + ay

            axes.annotate('', xy=(dx0, dy0), xytext=(dx1, dy1),
                    arrowprops=dict( headwidth=ahw, headlength=ahl, ec='royalblue', fc='royalblue', zorder=1))

##################################################################################
### -------------------------- MAIN SIMULATION LOOP -------------------------- ###
##################################################################################

    def start(self):

        ### ---- INITIALIZATION ---- ###

        start_time = datetime.now()
        self.create_agents()
        self.create_resources()
        self.stats, self.stats_pos = self.create_vis_field_graph()
        # turned_on_vfield = 0 # local var to decide when to show visual fields 

        if self.plot_trajectory: # start lists to track agent positional/mode activity
            traj_explore, traj_exploit, traj_collide = [],[],[]

        while self.t < self.T:

            if not self.is_paused:

                ### ---- OBSERVATIONS ---- ###

                for agent in self.agents:

                    ## Agent visual field projections
                    crash = agent.visual_sensing() # --> update vis_field
                    if crash: 
                        pygame.quit()
                        return 0, 0, True
                    ## Agent-wall collisions
                    agent.wall_contact_sensing() # --> update contact_field

                    ## Zero on_resrc parameter from previous step
                    agent.on_resrc = 0

                    ## Track agent activity (position + mode)
                    if self.plot_trajectory: 
                        x,y = agent.pt_center
                        if agent.mode == 'explore':
                            traj_explore.append([x, self.y_max - y + self.y_min])
                        elif agent.mode == 'exploit':
                            traj_exploit.append([x, self.y_max - y + self.y_min])
                        elif agent.mode == 'collide':
                            traj_collide.append([x, self.y_max - y + self.y_min])

                ## Agent-Resource collisions

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
                        
                ## Agent-Agent collisions
                
                # if self.collide_agents:
                        # Create dict of every agent that has collided : [colliding agents]
                    # collision_group_aa = pygame.sprite.groupcollide(self.agents, self.agents, False, False,
                    #     itra.within_group_collision)
                    #     # Carry out agent-agent collisions + generate list of non-exploiting collided agents
                    # for agent1, other_agents in collision_group_aa.items():
                    #     collided_agents_instance = self.agent_agent_collision_proximity(agent1, other_agents)
                    #     collided_agents.append(collided_agents_instance)
                    # collided_agents = [agent for sublist in collided_agents for agent in sublist]
                    # # Turn off collision mode when over
                    # for agent in self.agents:
                    #     if agent not in collided_agents and agent.get_mode() == "collide":
                    #         agent.set_mode("explore")

                ### ---- VISUALIZATION ---- ###

                for agent in self.agents:
                        agent.draw_update() # update agent color/position

                if self.with_visualization:
                    self.draw_frame(self.stats, self.stats_pos)
                    pygame.display.flip()

                ### ---- MODEL + ACTIONS ---- ###

                for agent in self.agents:

                    ## Pass observables through neural network to calculate actions (dvel + dtheta)
                    NN_input = agent.assemble_NN_inputs()
                    actions, agent.hidden = agent.NN.forward(NN_input, agent.hidden)

                    ## Consumption
                    if agent.on_resrc == 1: # --> consume upon collision
                    # if agent.on_resrc == 1 and agent.velocity == 0: # --> agent must learn to read on_resrc signal
                        self.consume(agent) # (sets agent.mode to exploit if still resources available)

                    ## Movement
                    else: agent.move(actions)

                # Step time forward
                self.t += 1

            # else: # simulation is paused
            #     # Still calculating visual fields
            #     for ag in self.agents:
            #         ag.calc_social_V_proj(self.agents)

            ### ---- BACKGROUND PROCESSES ---- ###

            # Carry out interaction according to user activity if not in headless mode
            if self.with_visualization:
                events = pygame.event.get() 
                self.interact_with_event(events)

            # # deciding if vis field needs to be shown in this timestep
            # turned_on_vfield = self.decide_on_vis_field_visibility(turned_on_vfield) ## turn off for speed

            # # Monitoring with IFDB (also when paused)
            # if self.save_in_ifd:
            #     ifdb.save_agent_data(self.ifdb_client, self.agents, self.t, exp_hash=self.ifdb_hash,
            #                          batch_size=self.write_batch_size)
            #     ifdb.save_resource_data(self.ifdb_client, self.resources, self.t, exp_hash=self.ifdb_hash,
            #                             batch_size=self.write_batch_size)
            # elif self.save_in_ram:
            #     # saving data in ram for data processing, only when not paused
            #     if not self.is_paused:
                    # ifdb.save_agent_data_RAM(self.agents, self.t)
            #         ifdb.save_resource_data_RAM(self.resources, self.t)

            # # Moving time forward
            # if (self.t % 500 == 0 or self.t == 1) and self.print_enabled:
            #     print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} t={self.t}")
            #     print(f"Simulation FPS: {self.clock.get_fps()}")
            self.clock.tick(self.framerate)

        ### ---- END OF SIMULATION ---- ###

        elapsed_time = datetime.now() - start_time
        # if self.print_enabled: print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} Total simulation time: ",
        #                             elapsed_time.total_seconds())

        # pop_num = None

        # if self.save_csv_files:
        #     if self.save_in_ifd or self.save_in_ram and self.print_enabled:
                # ifdb.save_ifdb_as_csv(exp_hash=self.ifdb_hash, use_ram=self.save_in_ram, as_zar=self.use_zarr,
                #                       save_extracted_vfield=False, pop_num=pop_num) # saves agent/res dict's into files
        #         env_saver.save_env_vars([self.env_path], "env_params.json", pop_num=pop_num)
        #     else:
        #         raise Exception("Tried to save simulation data as csv file due to env configuration, "
        #                         "but IFDB/RAM logging was turned off. Nothing to save! Please turn on IFDB/RAM logging"
        #                         " or turn off CSV saving feature.")

        # end_save_time = datetime.now()
        # if self.print_enabled: print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} Total saving time:",
        #                             elapsed_time.total_seconds())

        pygame.quit()

        fitness = [ag.collected_r for ag in self.agents][0]
        
        if self.print_enabled:
            print(fitness, elapsed_time.total_seconds())
        
        if self.plot_trajectory:
            self.plot_map(traj_explore, traj_exploit, traj_collide, w=4, h=4)

        return fitness, elapsed_time.total_seconds(), False