
import numpy as np
import matplotlib.pyplot as plt


x = np.random.normal(0,1,[1000])
y = x**2 + x + np.random.normal(0,0.1,[1000])

def main():
    plt.scatter(x,y)
    plt.show()

if __name__ == "__main__":
    main()