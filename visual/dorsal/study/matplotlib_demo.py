
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib


#%%
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


def init():
    line.setdata([], [])
    return line,


def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.setdata(x, y)
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200,
                               interval=20, blit=True)
anim.save('../result/study/sin.gif', fps=75, writer='imagemagick')
plt.show()
