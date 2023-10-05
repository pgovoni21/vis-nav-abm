import pygame
from abm import colors

class Wall(pygame.sprite.Sprite):

    def __init__(self, id, size, position, window_pad):
        super().__init__()

        # Saved parameters
        self.id = id

        # Visualization
        self.color = colors.GREY
        self.coll_color = colors.DARK_GREY

        self.image = pygame.Surface(size)
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = position + window_pad
