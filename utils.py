import numpy as np
import noise
import matplotlib.pyplot as plt

AX = np.newaxis

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

def plot_training(reward, loss):
	plt.style.use('seaborn-darkgrid')
	fig = plt.figure(figsize=(100, 300))
	ax = fig.add_subplot(211)
	ax.plot(reward, color='blue')
	ax.set(title="Mean reward per episode",
		   ylabel="Reward",
		   xlabel="Epoch")

	bx = fig.add_subplot(212)
	bx.plot(loss, color='red')
	bx.set(title="Mean loss per episode",
		   ylabel="Loss",
		   xlabel="Epoch")
	plt.show()