import pygame

class RenderPygame():
    def __init__(self, max_width, max_height):
        self.max_width = max_width
        self.max_height = max_height
        self.block_size = 640/self.max_width
        self.colour_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "yellow": (255, 255, 0),
            "purple": (255, 0, 255),
        }

        self.obj_colours = {
            "None": "red",
            "0": "purple",
            "1": "blue"
        }

        self.agent_colours = {
            "baseline": "red",
            "egalitarian": "yellow",
            "maximin": "blue",
            "utilitarian": "green",
            "berry": "purple"
        }
        self.screen = self.init_pygame()

    def init_pygame(self):
        # create window
        pygame.init()
        # setup screen
        width, height = self.max_width * self.block_size, self.max_height * self.block_size
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Harvest Model')
        # setup timer
        pygame.time.Clock()
        return screen

    def render_pygame(self, modelInst):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        self.screen.fill(self.colour_map["black"])
        for a in modelInst.schedule.agents:
            # if a has attribute deceased
            if hasattr(a, "off_grid") and a.off_grid:
                continue
            x = a.pos[0] * self.block_size
            y = a.pos[1] * self.block_size
            if a.agent_type == "berry":
                colour = self.colour_map[self.agent_colours[str(a.agent_type)]]
                self.draw_berry(self.screen, colour, x, y)
            else:
                colour = self.colour_map[self.agent_colours[str(a.agent_type)]]
                self.draw_agent(self.screen, colour, x, y)
        pygame.display.flip()
        return self.screen

    def draw_agent(self, screen, colour, x, y):
        pygame.draw.rect(screen, colour, (x+self.block_size/4, y, self.block_size/2, self.block_size/4))
        pygame.draw.circle(screen, "blue", (x+self.block_size/2-self.block_size/8,y+self.block_size/8), (self.block_size/8))
        pygame.draw.circle(screen, "blue", (x+self.block_size/2+self.block_size/8,y+self.block_size/8), (self.block_size/8))
        pygame.draw.rect(screen, colour, (x, y+self.block_size/4, self.block_size, self.block_size/4))
        pygame.draw.rect(screen, colour, (x, y+self.block_size/2, self.block_size/4, self.block_size/2))
        pygame.draw.rect(screen, colour, (x+self.block_size*3/4, y+self.block_size/2, self.block_size/4, self.block_size/2))

    def draw_berry(self, screen, colour, x, y):
        pygame.draw.circle(screen, colour, (x+self.block_size/2,y+self.block_size/2), (self.block_size/3))
        pygame.draw.arc(screen, "green", (x,y, self.block_size/2, self.block_size/2), 0/57, 90/57)