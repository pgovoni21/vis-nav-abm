def abs_color_to_float(color_tuple):
    """Given a color_tuple with R, G and B values between 0 and 255 we
    give back a touple with values between 0 and 1"""
    return tuple([comp/255 for comp in color_tuple])


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

GREY = (230, 230, 230)
DARK_GREY = (210, 210, 210)

RED = (184, 0, 0)
GREEN = (26, 158, 26)  # (50, 150, 50)
BLUE = (0, 109, 219)  # (0, 100, 255)

TOMATO = (255, 99, 71)
LIME = (50, 205, 50)
CORN = (100, 149, 237)
GOLD = (255, 215, 0)
VIOLET = (154, 14, 234)