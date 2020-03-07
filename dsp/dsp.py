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
    if b.size > 1:
        b1 = -b[1:]
        # print(b1)
        # print(b)
        # print(f"b: {b}, b1: {b1}")
        out = np.zeros(signal.shape)
        for n in range(signal.size):
            y_n = np.roll(out[-2::-1], n)[0:b1.size]
            # print(f"out       {n}: {out}")
            # print(f"out     rd{n}: {out[-2::-1]}")
            # print(f"out randrd{n}: {np.roll(out[-2::-1], n+1)}")
            # print(f"b1: {b1}")
            # print(f"y_{n}: {y_n}")
            # print(f"<b1,y_n>: {np.vdot(b1,y_n)}")
            out[n] = (np.vdot(b1,y_n) + v[n])/b[0]
    else:
        out = v/b
    return out

def resample(signal, N, M, kernelWidth=0): # resize by N/M
    # first, interpolate:
    bigger = np.zeros(N*signal.shape[0])
    for k in range(signal.size):
        bigger[N*k] = signal[k]
    # then filter:
    # first work out what the bandwidth of the filter should be:
    if N > M: # if the line is bigger than the original, restrict to the original bandwidth 
        kernel = sinc_kernel(1/N, kernelWidth)
    else: # if it's smaller than the original, restrict to the new bandwidth
        kernel = M*sinc_kernel(1/M, kernelWidth)
    bigger = fir_filt(bigger, N*kernel, kernelWidth)
    # then decimate:
    smaller = np.zeros(bigger.size//M) # floor division
    for k in range(smaller.size):
        smaller[k] = bigger[M*k]
    return smaller

# Synthesis

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

def FT(x, plot=False, orthonormal=False):
    if orthonormal:
        norm = len(x)**0.5
    else:
        norm = 1
    X = np.fft.fft(x)/norm
    w = np.linspace(0, 2.0, len(x))
    if plot:
        plt.subplot(211)
        plt.ylabel('Magnitude')
        plt.plot(w, np.abs(X))
        plt.subplot(212)
        plt.xlabel('Frequency (π rad/sample)')
        plt.ylabel('Phase (rad)')
        plt.plot(w, np.angle(X))
        plt.show()
    return [X, w]

# Test
def test_fir(): # need to figure out a metric
    N = 1024
    h = sinc_kernel(0.5,32)
    x = impulse(N)
    [X,_] = FT(x)
    y = fir_filt(x,h)
    [Y,w] = FT(y)
    hff = np.zeros(N)
    hff[0:h.size] = h
    [H,_] = FT(hff)

    plt.subplot(211)
    plt.ylabel('Magnitude')
    plt.plot(w, np.abs(Y/X))
    plt.plot(w, np.abs(H))
    plt.subplot(212)
    plt.xlabel('Frequency (π rad/sample)')
    plt.ylabel('Phase (rad)')
    plt.plot(w, np.angle(Y/X))
    plt.plot(w, np.angle(H))
    plt.show()

def test_iir():
    N = 1024
    a = np.array([1,])
    b = np.array([1,0,0.25])

    x = impulse(N)
    y = iir_filt(x, a, b)

    aff = np.zeros(N)
    bff = np.zeros(N)
    aff[0:a.size] = a
    bff[0:b.size] = b


    [A,w] = FT(aff)
    [B,_] = FT(bff)
    [X,_] = FT(x)
    [Y,_] = FT(y) 

    plt.subplot(211)
    plt.ylabel('Magnitude')
    plt.plot(w, np.abs(Y/X))
    plt.plot(w, np.abs(A/B))
    plt.subplot(212)
    plt.xlabel('Frequency (π rad/sample)')
    plt.ylabel('Phase (rad)')
    plt.plot(w, np.angle(Y/X))
    plt.plot(w, np.angle(A/B))
    plt.show()


def main():
    test_fir()
    test_iir()

if __name__ == "__main__":
    main()
