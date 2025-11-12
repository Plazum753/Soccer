import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(n_tir):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('nombre de tir')
    plt.plot(n_tir,"g*", label="nombre de tir")
    plt.xlim(left=0)
    plt.text(len(n_tir)-1,n_tir[-1],str(n_tir[-1]))

