import numpy as np
import matplotlib.pyplot as plt

def sinc_kernel(cutoff, halflength): # cutoff should be between 0 and 1, TODO window function, low-pass/highpass
    n = np.arange(-halflength, halflength + 1)
    # get the sinc function:
    out = np.sinc(cutoff*n)
    # return the normalised kernel
    return out/np.sum(out)

def filt(signal, kernel, centre=0): # not dealing with edge effects at all
    out = np.convolve(signal, kernel)
    return out[centre:centre+signal.shape[0]]

def resample(signal, N, M, kernelWidth=0): # resize by N/M
    # first, interpolate:
    bigger = np.zeros(N*signal.shape[0])
    for k in range(signal.shape[0]):
        bigger[N*k] = signal[k]
    # then filter:
    # first work out what the bandwidth of the filter should be:
    if N > M: # if the line is bigger than the original, restrict to the original bandwidth 
        kernel = sinc_kernel(1/N, kernelWidth)
    else: # if it's smaller than the original, restrict to the new bandwidth
        kernel = sinc_kernel(1/M, kernelWidth)
    bigger = filt(bigger, N*kernel, kernelWidth)
    # then decimate:
    smaller = np.zeros(bigger.shape[0]//M) # floor division
    for k in range(smaller.shape[0]):
        smaller[k] = bigger[M*k]
    return smaller

def main():
    fs1 = 1000
    fs2 = 2000
    x = np.arange(100)
    y = np.sin(2*np.pi*500*x/fs2)
    y2 = resample(y,2,1, 10)
    #y2 = sinc_kernel(0.1, 20)
    plt.plot(y)
    plt.plot(y2)
    plt.show()

if __name__ == "__main__":
    main()
