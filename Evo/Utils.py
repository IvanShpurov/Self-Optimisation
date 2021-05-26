import numpy as np 
def rgb_cycle(colors=4001):
    phi = np.linspace(0, 2*np.pi, 4001)
    x = np.sin(phi)
    y = np.cos(phi)
    rgb_cycle = np.vstack((            # Three sinusoids
        .5*(1.+np.cos(phi          )), # scaled to [0,1]
        .5*(1.+np.cos(phi+2*np.pi/3)), # 120° phase shifted.
        .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (60,3)
    return rgb_cycle