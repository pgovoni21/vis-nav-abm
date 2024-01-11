import pygame
import numpy as np
from abm import colors

class Landmark(pygame.sprite.Sprite):

    def __init__(self, id, color, radius, position, window_pad):
        super().__init__()

        # Saved parameters
        self.id = id
        self.color = color
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)

        # Visualization
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(self.color)
        self.image.set_colorkey(self.color)
        pygame.draw.circle(
            self.image, self.color, (radius, radius), radius
        )
        self.rect = self.image.get_rect(center = self.position + window_pad)
