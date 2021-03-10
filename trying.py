import numpy as np 
import matplotlib.pyplot as plt 

t = np.linspace(0.0, 2.0, 201) 
s = np.sin(2 * np.pi * t) 
   
fig, [ax, ax1] = plt.subplots(2, 1) 
   
ax.set_ylabel('y-axis') 
ax.plot(t, s) 
ax.grid(True) 
ax.set_title('Sample Example', 
             fontsize = 12, 
             fontweight ='bold') 
   
ax1.set_ylabel('y-axis') 
ax1.plot(t, s) 
ax1.grid(True) 
  
fig.clear(False) 
   
fig.suptitle('matplotlib.figure.Figure.clear()', fontweight ="bold") 
  
plt.show() 