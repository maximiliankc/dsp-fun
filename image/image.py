import argparse
from matplotlib import pyplot as plt
from matplotlib import image as mplImage
import numpy as np

class image:
    def __init__(self, inputPhoto):
        image = plt.imread(inputPhoto)
        self.info = {}
        if image.ndim == 3:
            (self.info['rows'], self.info['columns'], self.info['channels']) = image.shape
        elif image.ndim == 2:
            (self.info['rows'], self.info['columns']) = image.shape
            self.info['channels'] = 1
        else:
            raise Exception('2D images only!')

        self.image = []
        print(self.info)
        for z in range(self.info['channels']):
            # separating into channels
            self.image.append(image[:,:,z])
    
    def save(self, outFile):
        outImage = self.image[0]
        for z in range(1, self.info['channels']):
            outImage = np.dstack((outImage, self.image[z]))
        print(outImage.shape)
        mplImage.imsave(outFile, outImage)    

    def interpolate_channel(self, channel, n, background):
        c = background*np.ones((n[0]*channel.shape[0],n[1]*channel.shape[1]), channel.dtype)
        for x in range(channel.shape[0]):
            for y in range(channel.shape[1]):
                c[n[0]*x,n[1]*y] = channel[x,y]
        return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inName', dest='inFile')
    parser.add_argument('--outName', dest='outFile')

    args = parser.parse_args()
    inFile = args.inFile
    outFile = args.outFile

    photo = image(inFile)
    photo.save(outFile)


if __name__ == "__main__":
    main()