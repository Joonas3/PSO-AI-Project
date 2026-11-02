import numpy as np

#seed for stable results
np.random.seed(42)

class PSO:
    def __init__(self, func, particles = 10, max_iterations=100, bounds=(-500, 500), w=0.3, c1=1, c2=2):
        #function to search for global minimum
        self.func = func

        #population size
        self.particles = particles

        #dimension size
        self.dimensions = 2

        #how many times to iterate
        self.max_iterations = max_iterations

        #bounds for coordinates in search space
        self.bounds = bounds

        #parameters for Particle Swarm Optimization
        self.w = w #inertia which keeps particles moving
        self.c1 = c1 #attraction to pb (personal best)
        self.c2 = c2 #attraction to gb (global best)

        #create random particle positions for every particle
        self.positions = np.random.uniform(
            bounds[0],
            bounds[1],
            size=(particles, self.dimensions)
        )

        #give small random speeds for particles
        speed_range = (bounds[1]-bounds[0]) * 0.1

        self.speeds = np.random.uniform(
            -speed_range,
            speed_range,
            size=(particles, self.dimensions)
        )

        #configure pb positions
        self.pb_positions = self.positions.copy() #at start each particles pb is its current position

        #calculate the given function (Banana or Schwefel) to every particle to get pb values
        self.pb_values = np.array([
            self.func(position)
            for position in self.positions
        ])

        #configure gb positions
        best_index = np.argmin(self.pb_values) #particle with best pb value

        self.gb_position = self.pb_positions[best_index].copy() #best particle position stored
        self.gb_value = self.pb_values[best_index].copy() #best particle value stored

        self.bests = [] #store best positions and values for animation

    #function for new speeds for particles
    def update_speed(self):

        #random value for each particle and coordinate
        r1 = np.random.random(
            size=(self.particles, self.dimensions)
        )
        r2 = np.random.random(
            size=(self.particles, self.dimensions)
        )

        #compute new speed with the PSO formula
        self.speeds = (
            self.w * self.speeds +
            self.c1 * r1 * (self.pb_positions - self.positions) + 
            self.c2 * r2 * (self.gb_position - self.positions)
        )

    #function for updating particle positions
    def update_positions(self):

        #move every particle according to its speed
        self.positions += self.speeds

        #keep every particle in search space boundaries
        self.positions = np.clip(
            self.positions,
            self.bounds[0],
            self.bounds[1]
        )

    #function for new pbs and gbs
    def update_best(self):

        #calculate the given function (Banana or Schwefel) at particles new position
        current_values = np.array([
            self.func(position)
            for position in self.positions
        ])

        #check for improvements in pbs
        improved = current_values < self.pb_values

        #update pb positions for improved particles
        self.pb_positions[improved] = self.positions[improved]

        #update pb function values for improved particles
        self.pb_values[improved] = current_values[improved]

        #find best pb from the swarm
        best_index = np.argmin(self.pb_values)

        #update gb value and position if new gb is found
        if self.pb_values[best_index] < self.gb_value:

            self.gb_value = self.pb_values[best_index]
            self.gb_position = (self.pb_positions[best_index].copy())

    #optimization function
    def optimize(self):

        #repeat PSO process for given iterations
        for iteration in range(self.max_iterations):
            self.bests.append(self.positions.copy()) #save current particles for visualization
            self.update_speed() #update movement
            self.update_positions() #move particles
            self.update_best() #check for better solution

            #print progress
            #print(
            #    f"iteration {iteration + 1}: "
            #    f"best value = {self.gb_value:.6f}, "
            #    f"best position = {self.gb_position}"
            #)

        #save final particle positions
        self.bests.append(self.positions.copy())

        #return best solution
        return self.gb_position, self.gb_value