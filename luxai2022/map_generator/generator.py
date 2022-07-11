import numpy as np
from scipy.ndimage import convolve, maximum_filter
from scipy.fft import dctn, idctn
from visualize import viz
from symnoise import SymmetricNoise, symmetrize

# TODO:
# - (Sub)classes > 4 different functions.
# - Seeding
# - Pass in a SymmetricNoise class to the various generators and only use that one
#       (creating SymmetricNoise takes awhile (3ms), so if you don't care about maps
#        all having the same noise, you can generate them much faster)

class GameMap(object):
    def __init__(self, rubble, ice, ore):
        self.rubble = rubble
        self.ice = ice
        self.ore = ore
        self.width, self.height = len(rubble[0]), len(rubble)

def cave(width, height, symmetry="vertical"):
    # Build the cave system
    r = np.random.randint(6, size=(height, width))
    r[r >= 1] = 1
    symmetrize(r, symmetry)

    for i in range(3):
        r = convolve(r, [[1]*3]*3, mode="constant", cval=0) // 6

    r = 1-r
    r += maximum_filter(r, size=4)

    # Make it a little noisy
    rubble = SymmetricNoise(symmetry=symmetry, width=width, height=height)() * 50 + 50
    rubble = np.floor(rubble + 0.5)
    rubble[r==1] //= 5
    rubble[r==0] //= 20

    symmetrize(rubble, symmetry)

    return GameMap(rubble, r*50, r*50)

def craters(width, height, symmetry="vertical"):
    min_craters = max(2, width*height // 1000)
    max_craters = max(4, width*height // 500)
    craters = np.random.randint(min_craters, max_craters+1)

    rubble = np.zeros((height, width))
    xx, yy = np.mgrid[:width, :height]
    noise = SymmetricNoise(symmetry=symmetry, width=width, height=height)
    crater_noise = noise(frequency=10) * 0.5 + 0.75
    circles = []
    for i in range(craters):
        x, y = np.random.randint(width), np.random.randint(height)
        r = np.random.randint(3, min(width, height)//4)
        c = (xx - x) **2 + (yy - y)**2
        rubble[c.T * crater_noise < r**2] += 1

    rubble = rubble.astype(float)
    symmetrize(rubble, symmetry)
    rubble = (np.minimum(rubble, 1) * 90 + 10) * noise(frequency=3)

    return GameMap(rubble, rubble, rubble)




def island(width, height, symmetry="vertical"):
    # TODO: Use a faster algorithm for island.

    r = np.random.randint(4, size=(height, width))
    r[r >= 1] = 1

    s = -1
    while np.sum(r==0) != s:
        s = np.sum(r==0)
        r = convolve(r, [[1]*3]*3, mode="constant", cval=1) // 6

    # Add noise to the island shapes.
    xx, yy = np.mgrid[:width, :height]
    noise_x = SymmetricNoise(symmetry=symmetry, width=width, height=height)(frequency=10) * 4 - 2
    noise_y = SymmetricNoise(symmetry=symmetry, width=width, height=height)(frequency=10) * 4 - 2 # Need to be careful when seeding this to not use same seed as noise_x.
    new_xx = xx + np.round(noise_x)
    new_xx = np.maximum(0, np.minimum(width-1, new_xx)).astype(int)
    new_yy = yy + np.round(noise_y)
    new_yy = np.maximum(0, np.minimum(height-1, new_yy)).astype(int)
    prev_r = r.copy()
    r[yy, xx] = r[new_yy, new_xx]

    symmetrize(r, symmetry)

    rubble = SymmetricNoise(symmetry=symmetry, width=width, height=height)(frequency=3) * 50 + 50
    rubble[r==0] //= 20

    return GameMap(rubble, prev_r*100, r*100)

def solve_poisson(f):
    """
    Solves the Poisson equation
        ∇²p = f
    using the finite difference method with Neumann boundary conditions.
    References: https://elonen.iki.fi/code/misc-notes/neumann-cosine/
                https://scicomp.stackexchange.com/questions/12913/poisson-equation-with-neumann-boundary-conditions
    """
    nx, ny = f.shape

    # Transform to DCT space
    dct = dctn(f, type=1)

    # Divide by magical factors
    cx = np.cos(np.pi * np.arange(nx) / (nx - 1))
    cy = np.cos(np.pi * np.arange(ny) / (ny - 1))
    f = np.add.outer(cx, cy) - 2

    np.divide(dct, f, out=dct, where=f != 0)
    dct[f == 0] = 0

    # Return to normal space
    potential = idctn(dct, type=1)
    return potential / 2

def mountain(width, height, symmetry="vertical"):
    f = np.zeros((height, width))

    # Sprinkle a few mountains on the map.
    min_mountains = max(2, width*height//750)
    max_mountains = max(4, width*height//375)
    mountains = np.random.randint(min_mountains, max_mountains+1)
    while mountains > 0:
        x, y = np.random.randint(width), np.random.randint(height)
        f[y][x] -= 1
        mountains -= 1

    symmetrize(f, symmetry)
    rubble = solve_poisson(f)
    rubble *= 1 + SymmetricNoise(symmetry=symmetry, width=width, height=height)(frequency=3)
    rubble -= np.amin(rubble)
    rubble /= np.amax(rubble)
    rubble = np.floor(100 / np.max(rubble) * rubble)
    return GameMap(rubble, rubble, rubble)

def random_map():
    width = height = np.random.randint(32, 64)
    map_type = np.random.choice(["cave", "craters", "island", "mountain"])
    symmetry = np.random.choice(["horizontal", "vertical", "rotational", "/", "\\"])
    return eval(map_type)(width, height, symmetry)


if __name__ == "__main__":
    game_map = random_map()
    viz(game_map)
    input()
