import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal

from dbManager import initDb
from signalUtils import getOverSampledSignal, getSignalWithLabelGenerator


def cwt(x):
    sNew = getOverSampledSignal(x, 8)
    width = 32
    widths = np.arange(1, width)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.plot()


generator = getSignalWithLabelGenerator(initDb)

NUMBER_OF_SAMPLES_PER_METHOD = 1
for x in range(0, NUMBER_OF_SAMPLES_PER_METHOD):
    sample, label = generator(x)
    i, q = sample[0], sample[1]
    cwt(i)
    plt.show()
    cwt(q)
    plt.show()
