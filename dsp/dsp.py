import numpy as np
import matplotlib.pyplot as plt

# W = J/sample

## Filtering

def sinc_kernel(cutoff, halflength): # cutoff should be between 0 and 1, TODO window function, low-pass/highpass
    n = np.arange(-halflength, halflength + 1)
    # get the sinc function:
    out = np.sinc(cutoff*n)
    # return the normalised kernel
    return out/np.sum(out)

def fir_filt(signal, kernel, centre=0): # not dealing with edge effects at all, chopping the edges off
    out = np.convolve(signal, kernel)
    return out[centre:centre+signal.shape[0]]

def iir_filt(signal, a, b): # not dealing with edge effects at all, chopping the edges off
    # y[n] = ((m=0->∞)Σa_m*x[n-m] + (m=1->∞)Σb_m*y[n-m])/b_0
    # v[n] = (a * x)[n]
    # y[n] = (v[n] + <y_n,b>)/b_0
    v = fir_filt(signal, a)
    b1 = b[1:]
    print(f"b: {b}, b1: {b1}")
    out = np.empty(signal.shape)
    for n in range(signal.size):
        y = np.zeros()
        out[n] = np.vdot(b1,[out[n-1:-1:0], np.zeros(b1.size-n)]) + v[n]
        


def resample(signal, N, M, kernelWidth=0): # resize by N/M
    iir_filt(np.array([1,0,0]), np.array([1, 2, 3]), np.array([1,2,3,4,5,6]))
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
    for k in range(smaller.size):
        smaller[k] = bigger[M*k]
    return smaller

#synthesis

def impulse(N):
    x = np.zeros(N)
    x[0] = 1
    return x

def white_noise(std, length):
    return np.random.normal(0, std, length)

# Analysis

def energy(x):
    # returns energy, average power
    E = np.vdot(x,x).real
    return [E, E/len(x)]

def psd(x, plot=False):
    X = np.fft.fft(x)/(len(x)**0.5)
    psd = (X*np.conj(X)).real
    w = np.linspace(0, 2.0, len(x))
    if plot:
        plt.ylabel('Magnitude (J/sample)')
        plt.xlabel('Frequency (π rad/sample)')
        plt.plot(w, psd)
        plt.show()
    return [psd, w]

def plotFT(x):
    X = np.fft.fft(x)/(len(x)**0.5) # I like an orthonormal fourier transform
    w = np.linspace(0, 2.0, len(x))
    plt.subplot(211)
    plt.ylabel('Magnitude')
    plt.plot(w, np.abs(X))
    plt.subplot(212)
    plt.xlabel('Frequency (π rad/sample)')
    plt.ylabel('Phase (rad)')
    plt.plot(w, np.angle(X))
    plt.show()
    return [X, w]

def main():
    #y = impulse(100)
    y = white_noise(1,100)
    y2 = resample(y, 2, 1, 10)
    print(f"length of y: {len(y)}, length of y2: {len(y2)}")
    [Y,_] = plotFT(y)
    print(f"original Energy: {energy(y)}, transformed: {energy(Y)}")
    [Y2,_] = plotFT(y2)
    print(f"original Energy: {energy(y2)}, transformed: {energy(Y2)}")
    psd(y,True)
    psd(y2,True)
    plt.plot(y)
    plt.plot(y2)
    plt.show()

if __name__ == "__main__":
    main()
