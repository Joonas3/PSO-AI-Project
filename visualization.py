import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
 
 
def animate_pso(pso, title, known_minimum=None):

    #get the data from the PSO object
    func = pso.func          #function (Schwefel or Banana)
    low, high = pso.bounds   #edges of the search space
    frames = pso.bests       #list of particle positions, one entry per iteration
 
    #draw the function as a coloured background
    #grid of 100 x 100, covering the search space
    xs = np.linspace(low, high, 100)
    ys = np.linspace(low, high, 100)
    X, Y = np.meshgrid(xs, ys)
 
    #calculate function value at every grid point
    Z = np.zeros_like(X)
    for i in range(100):
        for j in range(100):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
 
    #log scale for colours
    Z = np.log10(1 + Z)
 
    #setup figure
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, Z, levels=40, cmap="viridis")  #coloured background
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
 
    #mark the real global minimum with a red star
    if known_minimum is not None:
        ax.plot(known_minimum[0], known_minimum[1], "r*", markersize=15,
                label="Global minimum")
 
    #particles: white dots
    dots, = ax.plot([], [], "wo", markeredgecolor="black", label="Particles")
 
    #text for the iteration number
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                   color="white", backgroundcolor="black")
 
    ax.legend(loc="lower right")
 
    #what to draw in each frame
    def update(frame_number):
        positions = frames[frame_number]                  #particles at this iteration
        dots.set_data(positions[:, 0], positions[:, 1])   #move the dots
        text.set_text(f"Iteration {frame_number}")        #update the text
        return dots, text
 
    #run animation 
    #show one frame per iteration and 200 ms between frames
    anim = FuncAnimation(fig, update, frames=len(frames), interval=200)
 
    plt.show()
    return anim  #keep reference so the animation stays running