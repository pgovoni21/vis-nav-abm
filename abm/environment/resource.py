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

    def __init__(self, id, radius, position, resrc_units=None, quality=1):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of Resource (int)
        :param radius: radius of the patch in pixels. This also refelcts the Resource units in the patch.
        :param position: position of the patch in env as (x, y)
        :param resrc_units: Resource units hidden in the given patch. If not initialized the number of units is equal to
                            the radius of the patch
        :param quality: quality of the patch in possibly exploitable units per timestep (per agent)
        """
        # PyGame Sprite superclass
        super().__init__()

        # Patch resource quantity
        if resrc_units is None:
            self.resrc_units = radius
        else:
            self.resrc_units = resrc_units

        # Saved parameters
        self.id = id
        self.radius = radius 
        self.resrc_left = self.resrc_units 
        self.position = position
        self.quality = quality 

        # Patch position
        self.center = (
            self.position[0] + self.radius, 
            self.position[1] + self.radius
            )

        # Visualization
        self.color = colors.GREY
        self.resrc_left_color = colors.DARK_GREY

        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, self.color, (radius, radius), radius
        )
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.center[0]
        self.rect.centery = self.center[1]

        self.show_stats = True # display resource information
        self.is_clicked = False # mouse events --> move patch + show_stats

### -------------------------- RESOURCE CONSUMPTION -------------------------- ###

    def deplete(self, consumption_rate):
        """
        Depleting the given patch with given Resource units
        """
        # Limits per agent consumption rate to that which the patch can provide
        if consumption_rate > self.quality:
            consumption_rate = self.quality

        # Steps resource unit counters if available
        if self.resrc_left >= consumption_rate:
            self.resrc_left -= consumption_rate
            depleted_units = consumption_rate
        else:  # self.resrc_left < consumption_rate --> agent consumes what is left
            depleted_units = self.resrc_left
            self.resrc_left = 0

        # Updates whether resources are left
        if self.resrc_left > 0:
            return depleted_units, False
        else:
            return depleted_units, True # --> kill agent

### -------------------------- VISUALIZATION / INTERACTION FUNCTIONS -------------------------- ###

    def draw_update(self): 
        """
        Drawing resource patch according to current state variables
        """
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
            text = font.render(f"{self.resrc_left:.2f}, Q{self.quality:.2f}", True, colors.BLACK)
            self.image.blit(text, (0, 0))
    
    def update_clicked_status(self, mouse):
        """
        Checking if the resource patch was clicked on a mouse event
        """
        if self.rect.collidepoint(mouse):
            self.is_clicked = True
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            self.center = (self.position[0] + self.radius, self.position[1] + self.radius)
            self.draw_update()
        else:
            self.is_clicked = False
            self.draw_update()