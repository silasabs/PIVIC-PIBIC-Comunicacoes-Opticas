import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import visualkeras
from PIL import ImageFont

def loadArray():


def plotIEEE_grid3x3(x, data, curves_numbers, curve_label, xlabel, ylabel, title, scale='dB'):
    
    plt.style.use(['science', 'ieee'])
    plt.figure(num=None, figsize=(14, 10))

    icon = ['-o', '-x', ]

    graph0 = plt.subplot(2, 1, 2)
    for index in range(0, curves_numbers - 1):
        plt.plot(x[index], data[index], icon[index], label=curve_label[index])
        plt.xlabel(xlabel[index])
        plt.ylabel(ylabel[index])
        plt.grid()
        plt.legend()
        plt.title(title)


def plotIEEE_grid2x2(x, data, xlabel, ylabel, title, scale='dB'):
    plt.style.use(['science', 'ieee'])

def plotIEEE_grid4x4(xlabel, ylabel, title, scale='dB'):
    plt.style.use(['science', 'ieee'])

def plotScience_grid3x3(xlabel, ylabel, title, scale='dB'):
    plt.style.use('science')

def plotScience_grid2x2(xlabel, ylabel, title, scale='dB'):
    plt.style.use('science')

def plotScience_grid4x4(xlabel, ylabel, title, scale='dB'):
    plt.style.use('science')

def plot3D(x, y, data, xlabel, ylabel, zlabel, title, scale='dB'):

    plt.style.use('science')

    x, y = np.meshgrid(x, y)
    
    if scale == 'dB':
        z = np.log10(data)
    else:
        z = data
    
    fig = plt.figure(figsize=(14, 8))
    ax =  plt.axes(projection='3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0.5, antialiased=True, edgecolor='None')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=7)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=30, azim=215)


def plotNN(path_image, model, font_size, NN='mlp'):

    font = ImageFont.truetype('arial.ttf', font_size)
    if NN == 'conv':      
        plot = visualkeras.layered_view(model, to_file=path_image+'/NN_plot.png', spacing=24, legend=True, font=font).show()
    else:
        plot = visualkeras.graph_view(model, to_file=path_image+'/NN_plot.png', legend='True', font=font).show()
    return plot