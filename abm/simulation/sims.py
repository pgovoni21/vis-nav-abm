import pygame
import numpy as np
import sys

from abm.agent import supcalc
from abm.agent.agent import Agent
from abm.environment.resource import Resource
from abm.contrib import colors, ifdb_params, evolution
from abm.simulation import interactions as itra
from abm.monitoring import ifdb
from abm.monitoring import env_saver
from math import atan2
import os
import uuid

import pprint
pp = pprint.PrettyPrinter(depth=4)
from datetime import datetime

def notify_agent(agent, status, res_id=None): 
    """Notifying agent  about the status of the environment in a given position"""
    agent.env_status_before = agent.env_status
    agent.env_status = status
    agent.novelty = np.roll(agent.novelty, 1)
    novelty = agent.env_status - agent.env_status_before
    novelty = 1 if novelty > 0 else 0
    agent.novelty[0] = novelty
    agent.pool_success = 1  # restarting pooling timer when notified
    if res_id is None:
        agent.exploited_patch_id = -1
    else:
        agent.exploited_patch_id = res_id


def refine_ar_overlap_group(collision_group): 
    """We define overlap according to the center of agents. If the collision is not yet with the center of agent,
    we remove that collision from the group"""
    for resrc, agents in collision_group.items():
        agents_refined = []
        for agent in agents:
            # Only keeping agent in collision group if its center is inside patch boundary
            # I.E: the agent can only get information from 1 point-like sensor in the center
            if supcalc.distance(resrc, agent) < resrc.radius:
                agents_refined.append(agent)
        collision_group[resrc] = agents_refined
    return collision_group


class Simulation:
    def __init__(self, N, T, field_res=800, width=600, height=480,
                 framerate=25, window_pad=30, with_visualization=True, show_vis_field=False,
                 show_vis_field_return=False, pooling_time=3, pooling_prob=0.05, agent_radius=10,
                 N_resrc=10, min_resrc_perpatch=200, max_resrc_perpatch=1000, min_resrc_quality=0.1, max_resrc_quality=1,
                 patch_radius=30, regenerate_patches=True, agent_consumption=1, teleport_exploit=True,
                 vision_range=150, agent_fov=1.0, visual_exclusion=False, show_vision_range=False,
                 use_ifdb_logging=False, use_ram_logging=False, save_csv_files=False, ghost_mode=True,
                 patchwise_exclusion=True, parallel=False, use_zarr=True, collide_agents=True):
        """
        Initializing the main simulation instance
        :param N: number of agents
        :param T: simulation time
        :param field_res: projection field (visual + proximity) resolution in pixels
        :param width: real width of environment (not window size)
        :param height: real height of environment (not window size)
        :param framerate: framerate of simulation
        :param window_pad: padding of the environment in simulation window in pixels
        :param with_visualization: turns visualization on or off. For large batch autmatic simulation should be off so
            that we can use a higher/maximal framerate.
        :param show_vis_field: (Bool) turn on visualization for visual field of agents
        :param show_vis_field_return: (Bool) sow visual fields when return/enter is pressed
        :param pooling_time: time units for a single pooling events
        :param pooling probability: initial probability of switching to pooling regime for any agent
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
        :param teleport_exploit: boolean to choose if we teleport agents to the middle of the res. patch during
                                exploitation
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
        :param ghost_mode: if turned on, exploiting agents behave as ghosts and others can pass through them
        :param patchwise_exclusion: excluding agents from social v field if they are exploiting the same patch as the
            focal agent
        :param parallel: if True we request to run the simulation parallely with other simulation instances and hence
            the influxDB saving will be handled accordingly.
        :param use_zarr: using zarr compressed data format to save single run data
        :param allow_border_patch_overlap: boolean switch to allow resource patches to overlap arena border
        :param agent_behave_param_list: list of dictionaries in which each dict is a copy of contrib.evolution.behave_params_template
            including the init parameters of all agents in case of heterogeneous agents.
        :param collide_agents: boolean switch agents can overlap if false.
        """

### -------------------------- INITIALIZATION -------------------------- ###

        # Arena parameters
        self.collide_agents = collide_agents
        self.WIDTH = width
        self.HEIGHT = height
        self.window_pad = window_pad

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

        # Visualization parameters
        self.show_vis_field = show_vis_field
        self.show_vis_field_return = show_vis_field_return
        self.show_vision_range = show_vision_range

        # Agent parameters
        self.agent_radii = agent_radius
        self.field_res = field_res
        self.pooling_time = pooling_time
        self.pooling_prob = pooling_prob
        self.agent_consumption = agent_consumption
        self.teleport_exploit = teleport_exploit # teleport_to_middle
        self.vision_range = vision_range
        self.agent_fov = agent_fov
        self.visual_exclusion = visual_exclusion
        self.ghost_mode = ghost_mode
        self.patchwise_exclusion = patchwise_exclusion

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

        # Initializing pygame
        if self.with_visualization:
            pygame.init()
            self.screen = pygame.display.set_mode([self.WIDTH + 2 * self.window_pad, self.HEIGHT + 2 * self.window_pad])
        else:
            pygame.display.init()
            pygame.display.set_mode([1,1])

        # pygame related class attributes
        self.agents = pygame.sprite.Group()
        self.resources = pygame.sprite.Group()
        self.clock = pygame.time.Clock() # todo: look into this more in detail so we can control dt

        # Monitoring
        self.use_zarr = use_zarr
        self.write_batch_size = None
        self.parallel = parallel
        if self.parallel:
            self.ifdb_hash = uuid.uuid4().hex
        else:
            self.ifdb_hash = ""
        self.save_in_ifd = use_ifdb_logging
        self.save_in_ram = use_ram_logging
        self.save_csv_files = save_csv_files
        if self.save_in_ram:
            self.save_in_ifd = False
            print("Turned off IFDB logging as RAM logging was explicitly requested!!!")

        if self.save_in_ifd:
            self.ifdb_client = ifdb.create_ifclient()
            if not self.parallel:
                self.ifdb_client.drop_database(ifdb_params.INFLUX_DB_NAME)
            self.ifdb_client.create_database(ifdb_params.INFLUX_DB_NAME)
            # ifdb.save_simulation_params(self.ifdb_client, self, exp_hash=self.ifdb_hash)
        else:
            self.ifdb_client = None

        # by default we parametrize with the .env file in root folder
        EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")

### -------------------------- DRAWING FUNCTIONS -------------------------- ###

    def draw_walls(self):
        """Drawing walls on the arena according to initialization, i.e. width, height and padding"""
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad],
                         [self.window_pad, self.window_pad + self.HEIGHT])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad],
                         [self.window_pad + self.WIDTH, self.window_pad])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad + self.WIDTH, self.window_pad],
                         [self.window_pad + self.WIDTH, self.window_pad + self.HEIGHT])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad + self.HEIGHT],
                         [self.window_pad + self.WIDTH, self.window_pad + self.HEIGHT])

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
        self.draw_walls()
        self.agents.draw(self.screen)
        if self.show_vision_range and self.WIDTH > self.vision_range: 
            self.draw_visual_fields()
        self.draw_framerate()
        self.draw_agent_stats()

        if self.show_vis_field:
            # showing visual fields of the agents
            self.show_visual_fields(stats, stats_pos)

    def draw_visual_fields(self):
        """Visualizing range of vision as opaque circles around the agents""" # todo --> fix limited FOV slice lines
        for agent in self.agents:
            # Show visual range 
            pygame.draw.circle(self.screen, colors.LIGHT_BLUE, agent.position + agent.radius, agent.vision_range, width=1)
            # Show limits of FOV 
            if self.agent_fov < np.pi:
                angles = [agent.orientation + agent.FOV[0], agent.orientation + agent.FOV[1]]
                for angle in angles: ### draws lines that don't quite meet borders
                    start_pos = (agent.position[0] + agent.radius, agent.position[1] + agent.radius)
                    end_pos = [start_pos[0] + (np.cos(angle)) * 3 * agent.radius,
                               start_pos[1] + (- np.sin(angle)) * 3 * agent.radius]
                    pygame.draw.line(self.screen, colors.LIGHT_BLUE,
                                     start_pos,
                                     end_pos, 1)

    def create_vis_field_graph(self):
        """Creating visualization graph for visual fields of the agents"""
        stats = pygame.Surface((self.WIDTH, 50 * self.N))
        stats.fill(colors.GREY)
        stats.set_alpha(150)
        stats_pos = (int(self.window_pad), int(self.window_pad))
        return stats, stats_pos

    def show_visual_fields(self, stats, stats_pos):
        """Showing visual fields of the agents on a specific graph"""
        stats_width = stats.get_width()
        # Updating our graphs to show visual field
        stats_graph = pygame.PixelArray(stats)
        stats_graph[:, :] = pygame.Color(*colors.WHITE)
        for k in range(self.N):
            show_base = k * 50
            show_min = (k * 50) + 23
            show_max = (k * 50) + 25

            for j in range(self.agents.sprites()[k].field_res):
                curr_idx = int(j * (stats_width / self.field_res))
                if self.agents.sprites()[k].soc_v_field[j] != 0:
                    stats_graph[curr_idx, show_min:show_max] = pygame.Color(*colors.GREEN)
                # elif self.agents.sprites()[k].soc_v_field[j] == -1:
                #     stats_graph[j, show_min:show_max] = pygame.Color(*colors.RED)
                else:
                    stats_graph[curr_idx, show_base] = pygame.Color(*colors.GREEN)

        del stats_graph
        stats.unlock()

        # Drawing
        self.screen.blit(stats, stats_pos)
        for agi, ag in enumerate(self.agents):
            line_height = 15
            font = pygame.font.Font(None, line_height)
            status = f"agent {ag.id}"
            text = font.render(status, True, colors.BLACK)
            self.screen.blit(text, (int(self.window_pad) / 2, self.window_pad + agi * 50))

### -------------------------- AGENT FUNCTIONS -------------------------- ###

    def create_agents(self):
        """Creating agents according to how the simulation class was initialized"""
        for i in range(self.N):
            # allowing agents to overlap arena borders (maximum overlap is radius of patch)
            x = np.random.randint(self.window_pad - self.agent_radii, self.WIDTH + self.window_pad - self.agent_radii)
            y = np.random.randint(self.window_pad - self.agent_radii, self.HEIGHT + self.window_pad - self.agent_radii)
            orient = np.random.uniform(0, 2 * np.pi) # randomly orients according to 0,pi/2 : right,up
            self.add_new_agent(i, x, y, orient)

    def add_new_agent(self, id, x, y, orient, with_prove=False): 
        """Adding a single new agent into agent sprites"""
        agent_proven = False
        while not agent_proven:
            agent = Agent(
                id=id,
                radius=self.agent_radii,
                position=(x, y),
                orientation=orient,
                env_size=(self.WIDTH, self.HEIGHT),
                color=colors.BLUE,
                field_res=self.field_res,
                FOV=self.agent_fov,
                window_pad=self.window_pad,
                pooling_time=self.pooling_time,
                pooling_prob=self.pooling_prob,
                consumption=self.agent_consumption,
                vision_range=self.vision_range,
                visual_exclusion=self.visual_exclusion,
                patchwise_exclusion=self.patchwise_exclusion
            )
            # if with_prove: ## with_prove is never True - left out for speed
            #     if self.prove_sprite(agent):
            #         self.agents.add(agent)
            #         agent_proven = True
            # else:
            #     self.agents.add(agent)
            #     agent_proven = True

            self.agents.add(agent)
            agent_proven = True

### -------------------------- RESOURCE FUNCTIONS -------------------------- ###

    def create_resources(self):
        """Creating resource patches according to how the simulation class was initialized"""
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
        
        resource_proven = False
        max_retries = 10000
        retries = 0
        while not resource_proven:
            if retries > max_retries:
                raise Exception("Reached timeout while trying to create resources without overlap!")
            
            radius = self.resrc_radius
            x = np.random.randint(self.window_pad, self.WIDTH + self.window_pad - 2 * radius)
            y = np.random.randint(self.window_pad, self.HEIGHT + self.window_pad - 2 * radius)

            units = np.random.randint(self.min_resrc_units, self.max_resrc_units)
            quality = np.random.uniform(self.min_resrc_quality, self.max_resrc_quality)
            resource = Resource(id, radius, (x, y), units, quality)

            # check for resource-resource overlap (does not check resource-agent overlap)
            resource_proven = self.prove_sprite(resource, prove_with_agents=False, prove_with_res=True)
            retries += 1

        self.resources.add(resource)

    def kill_resource(self, resource):
        """Killing (and regenerating) a given resource patch"""
        resource.kill()
        if self.regenerate_resources:
            self.add_new_resource_patch(force_id=resource.id)
    
### -------------------------- ABM INTERACTION FUNCTIONS -------------------------- ###

    def prove_sprite(self, sprite, prove_with_agents=True, prove_with_res=True):
        """Checks if proposed agent or resource is valid according to agent/patch overlap + returns True if collision,
        checking for agents/resources can be turned off with prove_with... parameters set to False"""
        
        if prove_with_res:
            collision_group_r = pygame.sprite.spritecollide(sprite, self.resources, False, pygame.sprite.collide_circle)
            if len(collision_group_r) > 0:
                return False
        
        # if prove_with_agents: # prove_with_agents is never True - left out for speed
        #     collision_group_a = pygame.sprite.spritecollide(sprite, self.agents, False, pygame.sprite.collide_circle)
        #     if len(collision_group_a) > 0: 
        #         return False
        
        return True

    def agent_agent_collision_proximity(self, agent1, other_agents):
        """
        Collision protocol called for agent-agent contact using proximity information to calculate turning behavior
        """
        collided_agents = []
        
        # does not apply for exploiting agents when ghost_mode is set to True
        if self.ghost_mode and agent1.get_mode() == "exploit":
            return collided_agents

        ### print("agent1 id/pos/orient: ", agent1.id, agent1.position, agent1.orientation)
        ### pp.pprint(agent1.field_obst_dict)

        for agent2 in other_agents:
            
            # skip loop iteration for exploiting agents when ghost_mode is set to True
            if self.ghost_mode and agent2.get_mode() == "exploit":
                continue

            # resetting agent mode
            if agent2.get_mode() != "exploit":
                agent2.set_mode("collide")

            # calculating proximity events
            # collecting agents closer than proximity sensor sensitivity
            agents_in_vicinity = [agent for agent in self.agents if
                                    supcalc.distance(agent, agent2) < 2 * agent2.radius + 20 and agent != agent2]
            
            # calculating proximity field similarly as LIDAR
            proximity_field = agent2.projection_field(agents_in_vicinity, keep_distance_info=True, fov=1)

            ### print("agent2 id/pos/orient: ", agent2.id, agent2.position, agent2.orientation)
            ### pp.pprint(agent2.field_obst_dict)

            # calculating turning angle
            V_field_len = len(proximity_field)
            left_excitation = np.mean(proximity_field[0:int(V_field_len / 2)])
            right_excitation = np.mean(proximity_field[int(V_field_len / 2)::])
            D_leftright = np.sign(left_excitation - right_excitation)
            if D_leftright == 0: D_leftright = -1
            if agent2.get_mode() != "exploit":
                # exploiting agents don't change orientation, non-exploiting agents turn away
                agent2.orientation -= D_leftright * 0.2

            # making agents stop if the front of the agent is occupied
            if np.mean(proximity_field[int(V_field_len / 2) - 100:int(V_field_len / 2) + 100]) > 0:
                agent2.velocity = 0
            else: # front is clear
                # making agent move straight if the front is clear
                if agent2.get_mode() != "exploit":
                    agent2.velocity = agent2.max_exp_vel
                    
            ### print("agent2 after collision: ", proximity_field, D_leftright, agent2.orientation)

            # add agent information to collided agent list
            collided_agents.append(agent2)

        return collided_agents

    def bias_agent_towards_res_center(self, agent, resrc, relative_speed=0.02):
        """Turning the agent towards the center of a resource patch with some relative speed"""
        x1, y1 = agent.position + agent.radius
        x2, y2 = resrc.center
        dx = x2 - x1
        dy = y2 - y1
        # calculating relative closed angle to agent2 orientation
        cl_ang = (atan2(dy, dx) + agent.orientation) % (np.pi * 2)
        agent.orientation += (cl_ang - np.pi) * relative_speed

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

##################################################################################
### -------------------------- MAIN SIMULATION LOOP -------------------------- ###
##################################################################################

    def start(self):

        ### ---- INITIALIZATION ---- ###

        start_time = datetime.now()
        print("Creating agents + resources + visual fields!")
        self.create_agents()
        self.create_resources()
        self.stats, self.stats_pos = self.create_vis_field_graph()

        # turned_on_vfield = 0 # local var to decide when to show visual fields ## turn off for speed

        print("Starting main simulation loop!")
        while self.t < self.T:

            # Carry out interaction according to user activity if not in headless mode
            if self.with_visualization:
                events = pygame.event.get() ## turn off for speed
                self.interact_with_event(events) ## turn off for speed

            # # deciding if vis field needs to be shown in this timestep
            # turned_on_vfield = self.decide_on_vis_field_visibility(turned_on_vfield) ## turn off for speed

            if not self.is_paused:

                ### ---- AGENT-AGENT INTERACTION ---- ###

                if self.collide_agents: ## set as True

                    # Check if any 2 agents has been collided and reflect them from each other if so
                    collision_group_aa = pygame.sprite.groupcollide(self.agents, self.agents, False, False,
                        itra.within_group_collision) # returns a dict (every agent that has collided : [colliding agents])

                    # Carry out agent-agent collisions + generate list of non-exploiting collided agents
                    collided_agents = []
                    for agent1, other_agents in collision_group_aa.items():
                        collided_agents_instance = self.agent_agent_collision_proximity(agent1, other_agents)
                        collided_agents.append(collided_agents_instance)
                    flat_list_collided_agents = [agent for sublist in collided_agents for agent in sublist]

                    # Turn off collision mode when over
                    for agent in self.agents:
                        if agent not in flat_list_collided_agents and agent.get_mode() == "collide":
                            agent.set_mode("explore")
                
                else:
                    collided_agents = []


                ### ---- AGENT-RESOURCE INTERACTION ---- ### (can not be separated from main thread for some reason)

                collision_group_ar = pygame.sprite.groupcollide(self.resources, self.agents, False, False,
                    pygame.sprite.collide_circle) # returns dict (every patch with agent : [foraging agents])

                # delete colliding agents if outside patch radius
                collision_group_ar = refine_ar_overlap_group(collision_group_ar)

                # collecting agents that are on resource patch
                agents_on_resrcs = []

                # Notifying agents about resource if pooling is successful + exploitation dynamics
                for resrc, agents in collision_group_ar.items():  # looping through patches
                    destroy_resrc = 0  # if we destroy a patch it is 1
                    for agent in agents:  # looping through all agents on patches
                        # Turn agent towards patch center
                        self.bias_agent_towards_res_center(agent, resrc)

                        # One of previous agents on patch consumed the last unit
                        if destroy_resrc:
                            notify_agent(agent, -1)
                        
                        # Agent finished pooling on a resource patch
                        if (agent.get_mode() in ["pool", "relocate"] and agent.pool_success) or agent.pooling_time == 0:
                            # Notify about the patch
                            notify_agent(agent, 1, resrc.id)
                            # Teleport agent to the middle of the patch if needed
                            # if self.teleport_exploit: ## turn off for speed
                            #     agent.position = resrc.position + resrc.radius - agent.radius

                        # Agent was already exploiting this patch
                        if agent.get_mode() == "exploit":
                            # continue depleting the patch
                            depl_units, destroy_resrc = resrc.deplete(agent.consumption)
                            agent.collected_r_before = agent.collected_r  # rolling resource memory
                            agent.collected_r += depl_units  # and increasing its collected resources
                            if destroy_resrc:  # consumed unit was the last in the patch
                                for agent_tob_notified in agents:
                                    notify_agent(agent_tob_notified, -1)

                        # Collect all agents on resource patches
                        agents_on_resrcs.append(agent)

                    # Patch is fully depleted
                    if destroy_resrc:
                        # we clear it from the memory and regenerate it somewhere else if needed
                        self.kill_resource(resrc)

                ### ---- NON-INTERACTING AGENTS ---- ###
                
                for agent in self.agents.sprites():
                    if agent not in agents_on_resrcs:  # for all the agents that are not on recourse patches
                        if agent not in collided_agents:  # and are not colliding with each other currently
                            # if they finished pooling
                            if (agent.get_mode() in ["pool",
                                                     "relocate"] and agent.pool_success) or agent.pooling_time == 0:
                                notify_agent(agent, -1)
                            elif agent.get_mode() == "exploit":
                                notify_agent(agent, -1)
                
                ### for agent in self.agents:
                ###     print("agent id/pos/orient: ", agent.id, agent.position, agent.orientation)
                ###     pp.pprint(agent.field_obst_dict)

                ### ---- GENERAL UPDATE PER TIME STEP ---- ###

                self.resources.update()
                self.agents.update(self.agents)
                self.t += 1

            # # Simulation is paused
            # else:
            #     # Still calculating visual fields
            #     for ag in self.agents:
            #         ag.calc_social_V_proj(self.agents)

            ### ---- BACKGROUND PROCESSES ---- ###

            # Draw environment and agents
            if self.with_visualization:
                self.draw_frame(self.stats, self.stats_pos)
                pygame.display.flip()

            # Monitoring with IFDB (also when paused)
            if self.save_in_ifd:
                ifdb.save_agent_data(self.ifdb_client, self.agents, self.t, exp_hash=self.ifdb_hash,
                                     batch_size=self.write_batch_size)
                ifdb.save_resource_data(self.ifdb_client, self.resources, self.t, exp_hash=self.ifdb_hash,
                                        batch_size=self.write_batch_size)
            elif self.save_in_ram:
                # saving data in ram for data processing, only when not paused
                if not self.is_paused:
                    ifdb.save_agent_data_RAM(self.agents, self.t)
                    ifdb.save_resource_data_RAM(self.resources, self.t)

            # Moving time forward
            if self.t % 500 == 0 or self.t == 1:
                print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} t={self.t}")
                print(f"Simulation FPS: {self.clock.get_fps()}")
            self.clock.tick(self.framerate)

        ### ---- END OF SIMULATION ---- ###

        end_time = datetime.now()
        print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} Total simulation time: ",
              (end_time - start_time).total_seconds())

        pop_num = None

        if self.save_csv_files:
            if self.save_in_ifd or self.save_in_ram:
                ifdb.save_ifdb_as_csv(exp_hash=self.ifdb_hash, use_ram=self.save_in_ram, as_zar=self.use_zarr,
                                      save_extracted_vfield=False, pop_num=pop_num)
                env_saver.save_env_vars([self.env_path], "env_params.json", pop_num=pop_num)
            else:
                raise Exception("Tried to save simulation data as csv file due to env configuration, "
                                "but IFDB/RAM logging was turned off. Nothing to save! Please turn on IFDB/RAM logging"
                                " or turn off CSV saving feature.")

        end_save_time = datetime.now()
        print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} Total saving time:",
              (end_save_time - end_time).total_seconds())

        pygame.quit()