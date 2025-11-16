import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(n_tir, n_touches):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Nombre de buts')
    plt.ylabel('nombre de tirs')
    plt.plot(n_tir,"g*", label="nombre de tirs")
    plt.plot(n_touches,"r*", label="nombre de touches")
    plt.xlim(left=0)
    plt.ylim(ymin=0)

