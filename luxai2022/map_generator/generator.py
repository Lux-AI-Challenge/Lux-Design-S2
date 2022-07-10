import numpy as np
from scipy.ndimage import convolve, maximum_filter
from scipy.fft import dctn, idctn
from visualize import viz
from symnoise import SymmetricNoise, symmetrize

class GameMap(object):
    def __init__(self, rubble, ice, ore):
        self.rubble = rubble
        self.ice = ice
        self.ore = ore
        self.width, self.height = len(rubble[0]), len(rubble)

def cave(width, height, symmetry="vertical"):
    r = np.random.randint(4, size=(height, width))
    r[r >= 1] = 1
    symmetrize(r, symmetry)

    for i in range(3):
        r = convolve(r, [[1]*3]*3, mode="constant", cval=0) // 6

    r = 1-r
    r += maximum_filter(r, size=5)

    rubble = np.random.randint(50, 100, size=r.shape)
    rubble[r==1] //= 10
    rubble[r==0] = 0

    symmetrize(rubble, symmetry)

    return GameMap(rubble, r*50, r*50)

def circle(x, y, r):
    # The points that form a closed circle around (x, y) of radius r.
    points = []
    c = (x+r, y) # Current point on circumference.
    r2 = r**2
    dist = r2 # Distance squared from center of circle.
    while True:
        points.append(c)

        # Find which quadrant we are in.
        if c[0] >= x and c[1] >= y:
            dx, dy = 1, -1
        elif c[0] >= x:
            dx, dy = -1, -1
        elif c[1] >= y:
            dx, dy = 1, 1
        else:
            dx, dy = -1, 1

        distx = dist + 1 + 2*dx*(c[0]-x)
        disty = dist + 1 + 2*dy*(c[1]-y)

        if abs(distx - r2) < abs(disty - r2):
            c = (c[0] + dx, c[1])
            dist = distx
        else:
            c = (c[0], c[1] + dy)
            dist = disty

        if c == (x+r, y):
            break
    return points


def craters(width, height, symmetry="vertical"):
    min_craters = max(2, width*height // 1000)
    max_craters = max(3, width*height // 500)
    craters = np.random.randint(min_craters, max_craters+1)

    rubble = np.zeros((height, width))
    circles = []
    while craters > 0:
        x, y = np.random.randint(width), np.random.randint(height)
        r = np.random.randint(3, min(width, height)//4)
        if any((c[0] - x)**2 + (c[1] - y)**2 < (c[2]+r)**2 for c in circles):
            continue

        circles.append((x, y, r))
        craters -= 1

    for c in circles:
        points = circle(*c)
        for x, y in points:
            if 0<= x < width and 0 <= y < height:
                rubble[y, x] = 100
    symmetrize(rubble, symmetry)

    return GameMap(rubble, rubble, rubble)




def island(width, height, symmetry="vertical"):
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
    map_type = "mountain"
    return eval(map_type)(width, height, symmetry)


if __name__ == "__main__":
    game_map = random_map()
    viz(game_map)
    input()
