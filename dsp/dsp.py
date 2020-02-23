import numpy as np
import matplotlib.pyplot as plt

def sinc_kernel(cutoff, halflength): # cutoff should be between 0 and 1, TODO window function, low-pass/highpass
    n = np.arange(-halflength, halflength + 1)
    # get the sinc function:
    out = np.sinc(cutoff*n)
    # return the normalised kernel
    return out/np.sum(out)

def fir_filt(signal, kernel, centre=0): # not dealing with edge effects at all, chopping the edges off
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
        kernel = M*sinc_kernel(1/M, kernelWidth)
    bigger = fir_filt(bigger, N*kernel, kernelWidth)
    # then decimate:
    smaller = np.zeros(bigger.shape[0]//M) # floor division
    for k in range(smaller.shape[0]):
        smaller[k] = bigger[M*k]
    return smaller

def impulse(N):
    x = np.zeros(N)
    x[0] = 1
    return x

def plotFT(x):
    X = np.fft.fft(x)/(len(x)**0.5)
    w = np.linspace(0, 2.0, len(x))
    plt.subplot(211)
    plt.plot(w, np.abs(X))
    plt.subplot(212)
    plt.plot(w, np.angle(X))
    plt.show()

def main():
    x = np.arange(100)
    y = impulse(100)
    y2 = resample(y, 1, 2, 10)
    print(f"length of y: {len(y)}, length of y2: {len(y2)}")
    plotFT(y)
    plotFT(y2)
    plt.plot(y)
    plt.plot(y2)
    plt.show()

if __name__ == "__main__":
    main()
