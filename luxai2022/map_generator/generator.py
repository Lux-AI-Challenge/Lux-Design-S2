import numpy as np
from scipy.ndimage import convolve, maximum_filter
from scipy.fft import dctn, idctn
from visualize import viz



class GameMap(object):
    def __init__(self, rubble, ice, ore):
        self.rubble = rubble
        self.ice = ice
        self.ore = ore
        self.width, self.height = len(rubble[0]), len(rubble)

def symmetrize(x, symmetry="vertical"):
    # Makes it match symmetry, in place.
    width, height = x.shape
    if symmetry == "horizontal":
        x[height//2:] = x[(height-1)//2::-1]
    elif symmetry == "vertical":
        x[:, width//2:] = x[:, (width-1)//2::-1]
    elif symmetry == "rotational":
        x[height//2:, :] = x[(height-1)//2::-1, ::-1]

def island(width, height, symmetry=None):
    if symmetry is None:
        symmetry = np.random.choice(["horizontal", "vertical", "rotational"])

    r = np.random.randint(4, size=(height, width))
    r[r >= 1] = 1
    symmetrize(r, symmetry)

    s = -1
    while np.sum(r==0) != s:
        s = np.sum(r==0)
        r = convolve(r, [[1]*3]*3, mode="constant", cval=1) // 6

    rubble = np.random.randint(50, 100, size=r.shape)
    rubble[r==0] //= 20
    
    symmetrize(rubble, symmetry)

    return GameMap(rubble, r*100, r*100)

def cave(width, height, symmetry=None):
    if symmetry is None:
        symmetry = np.random.choice(["horizontal", "vertical", "rotational"])

    iters = 3
    r = np.random.randint(4, size=(height, width))
    r[r >= 1] = 1
    symmetrize(r, symmetry)

    for i in range(iters):
        r = convolve(r, [[1]*3]*3, mode="constant", cval=0) // 6

    r = 1-r
    r += maximum_filter(r, size=5)
    
    rubble = np.random.randint(50, 100, size=r.shape)
    rubble[r==1] //= 10
    rubble[r==0] = 0
    
    symmetrize(rubble, symmetry)

    return GameMap(rubble, r*50, r*50)

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

def mountain(width, height, symmetry=None):
    if symmetry is None:
        symmetry = np.random.choice(["horizontal", "vertical", "rotational"])

    f = np.zeros((height, width))
    
    # Sprinkle a few mountains on the map.
    min_mountains = max(1, width*height//1000)
    max_mountains = max(3, width*height//500)
    mountains = np.random.randint(min_mountains, max_mountains)
    while mountains > 0:
        x, y = np.random.randint(width), np.random.randint(height)
        if symmetry == "vertical":
            if x >= width // 2:
                continue
        else:
            if y >= height // 2:
                continue
        f[y][x] -= 1
        mountains -= 1
        
    symmetrize(f, symmetry)
    rubble = solve_poisson(f)
    rubble -= np.amin(rubble)
    rubble /= np.amax(rubble)
    rubble **= 2
    rubble = np.floor(100 * rubble)
    return GameMap(rubble, rubble, rubble)

if __name__ == "__main__":
    game_map = mountain(64, 64)
    viz(game_map)
