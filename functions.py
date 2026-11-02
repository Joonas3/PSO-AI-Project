import numpy as np

# Banana function in 2D
def banana(position):
    x = position[0]
    y = position[1]

    return (1-x)**2 + 100*(y - x*x)**2

# Schwefel function in 2D
def schwefel(position):
    d = 2
    total = 0

    for i in range(d):
        xi = position[i]
        total += xi*np.sin(np.sqrt(abs(xi)))

    return 418.9829*d-total