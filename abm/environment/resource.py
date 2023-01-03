"""
Resource.py : including the main classes to create a Resource entity that can be exploited by agents.
"""
import pygame
import numpy as np
from abm.contrib import colors


class Resource(pygame.sprite.Sprite):
    """
        Resource class that includes all private parameters of the Resource patch and all methods necessary to exploit
        the Resource and change the patch size/appearance accordingly
        """

    def __init__(self, id, radius, position, env_size, color, window_pad, resrc_units=None, quality=1):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of Resource (int)
        :param radius: radius of the patch in pixels. This also refelcts the Resource units in the patch.
        :param position: position of the patch in env as (x, y)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the patch as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels
        :param resrc_units: Resource units hidden in the given patch. If not initialized the number of units is equal to
                            the radius of the patch
        :param quality: quality of the patch in possibly exploitable units per timestep (per agent)
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # Deciding how much resrc. is in patch
        if resrc_units is None:
            self.resrc_units = radius
        else:
            self.resrc_units = resrc_units

        # Initializing agents with init parameters
        self.id = id
        self.radius = radius  # saved
        self.resrc_left = self.resrc_units  # saved
        self.position = np.array(position, dtype=np.float64)  # saved
        self.center = (self.position[0] + self.radius, self.position[1] + self.radius)
        self.color = color
        self.resrc_left_color = colors.DARK_GREY
        self.unit_per_timestep = quality  # saved
        self.is_clicked = False
        self.show_stats = True

        # Environment related parameters
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.window_pad = window_pad
        self.boundaries_x = [self.window_pad, self.window_pad + self.WIDTH]
        self.boundaries_y = [self.window_pad, self.window_pad + self.HEIGHT]

        # Initial Visualization of Resource
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, self.color, (radius, radius), radius
        )
        # visualizing left Resources
        small_radius = int((self.resrc_left / self.resrc_units) * self.radius)
        pygame.draw.circle(
            self.image, self.resrc_left_color, (self.radius, self.radius), small_radius
        )
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.center[0]
        self.rect.centery = self.center[1]
        if self.is_clicked:
            font = pygame.font.Font(None, 25)
            text = font.render(f"{self.radius}", True, colors.BLACK)
            self.image.blit(text, (0, 0))

    def update_clicked_status(self, mouse):
        """Checking if the resource patch was clicked on a mouse event"""
        if self.rect.collidepoint(mouse):
            self.is_clicked = True
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            self.center = (self.position[0] + self.radius, self.position[1] + self.radius)
            self.draw_update()
        else:
            self.is_clicked = False
            self.draw_update()

    def draw_update(self): 
        """Drawing resource patch according to current state variables"""
        # Initial Visualization of Resource
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, self.color, (self.radius, self.radius), self.radius
        )
        small_radius = int((self.resrc_left / self.resrc_units) * self.radius)
        pygame.draw.circle(
            self.image, self.resrc_left_color, (self.radius, self.radius), small_radius
        )
        self.rect = self.image.get_rect()
        self.rect.centerx = self.center[0]
        self.rect.centery = self.center[1]
        self.mask = pygame.mask.from_surface(self.image)
        if self.is_clicked or self.show_stats:
            font = pygame.font.Font(None, 18)
            text = font.render(f"{self.resrc_left:.2f}, Q{self.unit_per_timestep:.2f}", True, colors.BLACK)
            self.image.blit(text, (0, 0))
            text_rect = text.get_rect(center=self.rect.center)

    def deplete(self, Resource_units):
        """depeting the given patch with given Resource units"""
        # Not allowing faster depletion than what the patch can provide (per agent)
        if Resource_units > self.unit_per_timestep:
            Resource_units = self.unit_per_timestep

        if self.resrc_left >= Resource_units:
            self.resrc_left -= Resource_units
            depleted_units = Resource_units
        else:  # can not deplete more than what is left
            depleted_units = self.resrc_left
            self.resrc_left = 0
        if self.resrc_left > 0:
            return depleted_units, False
        else:
            return depleted_units, True