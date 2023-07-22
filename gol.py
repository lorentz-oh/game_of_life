import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy

def next_gen(world: np.ndarray) -> np.ndarray:
    #neihgbour kernel
    nbor_kern = np.ones((3,3), np.uint8)
    nbor_kern[1,1] = 0
    #number of neighbours
    num_of_nbors = scipy.signal.convolve2d(world, nbor_kern, mode="same")
    world_next = np.zeros(world.shape, world.dtype)
    world_next[num_of_nbors==3] = 1
    world_next[num_of_nbors==2] = world[num_of_nbors==2]
    return world_next

def get_rand_world(size) -> np.ndarray:
    world = np.zeros((size,size), dtype=np.uint8)
    rng = np.random.default_rng()
    a,b = int(size*0.25), int(size*0.75)
    world[a:b, a:b] = rng.integers(2,size=(b-a, b-a))
    return world

if __name__ != "__main__":
    exit()

world = get_rand_world(200)

fig = plt.figure()

im = plt.imshow(world, animated=True)

def updatefig(*args):
    global world
    world = next_gen(world)
    im.set_array(world)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()