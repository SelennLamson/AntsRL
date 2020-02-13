import numpy as np
import noise

def perlin_noise_generator(w, h, offset_x, offset_y, scale=22.0, octaves=2, persistence=0.5, lacunarity=2.0):
	shape = (w, h)
	gen = np.zeros(shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			gen[i][j] = noise.pnoise2((i + offset_x) / scale,
									  (j + offset_y) / scale,
								      octaves=octaves,
								      persistence=persistence,
								      lacunarity=lacunarity,
								      base=0)
	return gen
