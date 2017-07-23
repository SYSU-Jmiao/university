import matplotlib.pyplot as plt

from dbManager import initDb
from signalUtils import getSignalWithLabelGenerator
from signalUtils import crossSpectrumDensity, spectogram, periodGram, cwt, stft

NUMBER_OF_SAMPLES_PER_METHOD = 6
NUMBER_OF_ROWS = 3
NUMBER_OF_COLUMNS = 2

generator = getSignalWithLabelGenerator(initDb)

for x in range(0, NUMBER_OF_SAMPLES_PER_METHOD):
    sample, label = generator(x)
    plt.subplot(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, x+1)
    plt.tight_layout()
    plt.title(label)
    crossSpectrumDensity(sample)

plt.show();

for process in [spectogram, periodGram, cwt, stft]:
    for x in range(0, NUMBER_OF_SAMPLES_PER_METHOD-1, 2):
        sample, label = generator(x)
        plt.subplot(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, x+1)
        plt.tight_layout()
        process(sample[0])
        plt.title( " I " + label)
        plt.subplot(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, x+2)
        plt.tight_layout()
        process(sample[1])
        plt.title( " Q " + label)

    plt.show();