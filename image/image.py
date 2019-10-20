import argparse
from matplotlib import pyplot as plt
from matplotlib import image as mplImage
import numpy as np

class image:

    # colour transform matrices
    Identity3 = np.eye(3)
    RGBtoBWNaive3 = np.ones((3,3))/3

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
        print(self.image[0].dtype)
        if self.image[0].dtype is np.dtype(np.uint8):
            for k in range(self.info['channels']):
                self.image[k] = self.image[k]/255.0
            print(self.image[0][50,50])
    
    def save(self, outFile):
        outImage = self.image[0]
        for z in range(1, self.info['channels']):
            outImage = np.dstack((outImage, self.image[z]))
        print(f'outshape: {outImage.shape}')
        mplImage.imsave(outFile, outImage)    

    def interpolate_image(self, n, background):
        new_image = []
        for k in range(self.info['channels']):
            new_image.append(self.interpolate_channel(self.image[k], n, background[k]))
        self.image = new_image

    def interpolate_channel(self, channel, n, background):
        c = background*np.ones((n[0]*channel.shape[0],n[1]*channel.shape[1]), channel.dtype)
        for x in range(channel.shape[0]):
            for y in range(channel.shape[1]):
                c[n[0]*x,n[1]*y] = channel[x,y]
        return c

    def gamma(self, gamma):
        for k in range(self.info['channels']):
            self.image[k]**gamma

    def colour_transform(self, C):
        # this is pretty bloody slow, should probably be optimised a lot, maybe recombine into a 3d np array?
        # assumes that C is an np array
        print(C)
        (newN, oldN) = C.shape
        print(f'{newN}, {oldN}')
        if oldN < self.info['channels']:
            print("assuming excess channels aren't colours") # discarding extra channels for now
        elif oldN > self.info['channels']:
            print("more colours than channels")
            return
        
        (x,y) = self.image[0].shape

        newImage = []
        for k in range(newN): # creating new image
            newImage.append(np.empty((x,y)))

        for k in range(x): # going through pixel by pixel
            for l in range(y):
                v = np.array([])
                for m in range(oldN):
                  v = np.append(v, self.image[m][k,l])
                newV = C@v #should be protected against dimensional mis-match (@ is matrix multiplication I believe)
                for m in range(newN):
                    newImage[m][k,l] = newV[m]
        self.info['channels'] = newN
        self.image = newImage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inName', dest='inFile')
    parser.add_argument('--outName', dest='outFile')

    args = parser.parse_args()
    inFile = args.inFile
    outFile = args.outFile

    photo = image(inFile)
    background = np.ones(photo.info['channels'])
    photo.interpolate_image([2,2], background)
    photo.save(outFile)


if __name__ == "__main__":
    main()