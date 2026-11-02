from pso import PSO
from functions import schwefel, banana
from visualization import animate_pso

#calculate Schwefel function global minimum
pso_schwefel = PSO(
    func=schwefel,
    particles=30,
    max_iterations=100,
    bounds=(-500, 500)
)

#optimize Schwefel function
best_position_schwefel, best_value_schwefel = pso_schwefel.optimize()

print("Best position Schwefel function:", best_position_schwefel)
print("Best value Schwefel function:", best_value_schwefel)

#calculate Banana function global minimum
pso_banana = PSO(
    func=banana,
    particles=30,
    max_iterations=100,
    bounds=(-2,2)
)

#optimize Banana function
best_position_banana, best_value_banana = pso_banana.optimize()

print("Best position for Banana function:", best_position_banana)
print("Best value for Banana function:", best_value_banana)

#animate both functions
anim1 = animate_pso(pso_schwefel, "Schwefel function",
                    known_minimum=(420.9687, 420.9687))

anim2 = animate_pso(pso_banana, "Banana function",
                    known_minimum=(1, 1))