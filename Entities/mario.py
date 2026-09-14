from Entities.entity import Entity

class Mario(Entity):
    def __init__(self):
        super().__init__()
        self.jump_height = .01
