import numpy as np
import matplotlib.pyplot as plt

print("Python is working!")
print("NumPy version:", np.__version__)

x = np.linspace(-5, 5, 100)
y = x**2

plt.plot(x, y)
plt.title("Python setup test")
plt.xlabel("x")
plt.ylabel("y")
plt.show()