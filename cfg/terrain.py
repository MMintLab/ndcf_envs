import numpy as np

class DiscreteTerrainConfig():
    def __init__(self):
        self.num_terrains = 1
        self.terrain_width = 0.55
        self.terrain_length = 0.55

        self.horizontal_scale = 0.001 # [m]
        self.vertical_scale = 0.001  # [m]

        self.max_height = 0.01
        self.min_size = 0.01
        self.max_size = 0.1
        self.num_rects = 60
        self.platform_size = 1.
        self.transform_xyz = np.array([-0.08, -0.08, 0.005])
        self.num_rows = int(self.terrain_width/self.horizontal_scale)
        self.num_cols = int(self.terrain_length/self.horizontal_scale)
        self.heightfield = np.zeros((self.num_terrains*self.num_rows, self.num_cols), dtype=np.int16)

        self.indent_distance = 0.01

class WaveTerrainConfig():
    def __init__(self):
        self.num_terrains = 1
        self.num_waves = np.random.uniform(1, 3)
        self.amplitude = 0.02

        self.terrain_width = 0.55
        self.terrain_length = 0.55
        self.horizontal_scale = 0.001  # [m]
        self.vertical_scale = 0.001  # [m]

        self.transform_xyz = np.array([-0.08, -0.08, 0.015])
        self.num_rows = int(self.terrain_width / self.horizontal_scale)
        self.num_cols = int(self.terrain_length / self.horizontal_scale)
        self.heightfield = np.zeros((self.num_terrains * self.num_rows, self.num_cols), dtype=np.int16)

        self.indent_distance =  0.015