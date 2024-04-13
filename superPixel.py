"""
    Second phase of Image Classification : Complex Network based approach .
    June 2022
"""
# import required packages 

import math
import numpy as np
from skimage import io, color
import timeit # DEBUG
import csv
import cv2


class Cluster(object):

    """
        Cluster data structure class used for organizing superpixel information

        Constructor Input:
            x: (Int)    Horizontal component of cluster center 
            y: (Int)    Vertical component of cluster center
            l: (Float)  Lightness component
            a: (Float)  Green-Red component
            b: (Float)  Blue-yellow component
    """

    # Static cluster index for labeling pixels of given image
    clusterIdx = 0

    def setCluster(self, x, y, r , g, b):

        self.x = x
        self.y = y
        self.r = r
        self.g = g
        self.b = b

    def __init__(self, x, y, r, g , b):

        self.setCluster(x, y,r, g , b)
        self.pixelsOfCluster = set()
        self.idx = Cluster.clusterIdx
        Cluster.clusterIdx += 1

class SLIC(object):

    """
    Processor class used to execute SLIC algorithm on a given image

    Constructor Input:
        Filepath: Name of file, or path to file
        K: (Int) Number of desired superpixels
        M: (Int) Compactness (scaling of distance)
    """


    def __init__(self, filepath, K = 10000, M = 10):

        """
        Input:
            Filepath: Name of file, or path to file
            K: (Int) Number of desired superpixels
            M: (Int) Compactness (scaling of distance)
        """

        # Initialize number of superpixels (K), and compactness (M)
        self.K = K
        self.M = M

        # Read in image from filepath as CIELAB color space ndarray 
        self.colorArr = color.rgb2lab(io.imread(filepath))
        self.imageArr = np.copy(self.colorArr)
        
        # Set dimensions
        self.height = self.colorArr.shape[0]
        self.width = self.colorArr.shape[1]

        # Set number of pixels (N), and superpixel interval size (S)
        self.N = self.height * self.width
        self.S = int(math.sqrt(self.N / K))

        # Track clusters, and labels for each pixel
        self.clusters = []
        self.labels = {}

        # Tracks distances to nearest cluster center (Initialized as largest possible value)
        self.distanceArr = np.full((self.height, self.width), np.inf)

    def execute(self, iterations, labWeight = 0.5, isBordered = True):

            """
                Perform SLIC on image given number of iterations and compactness value

                Input:
                    iterations - (Int)   Number of iterations to perform SLIC
                    labWeight -  (Float) Value between 0.0 and 1.0 to adjust effectiveness of LAB distance during labeling
                    isBordered - (Bool)  Boolean value representing if output will have borders around clusters
            """

            self.initializeClusters()
            self.updateClusters()

            for i in range(iterations):

                start = timeit.default_timer()
                self.labelPixels(labWeight)
                self.updateCenters()
                self.enforceConnectivity()
                name = '/home/asma/Documents/Programing/BachelorProject/bachelorPro/Bachelor-Project/images/test_M{m}_K{k}_loop{loop}.png'.format(loop = i, m = self.M, k = self.K)
                stop = timeit.default_timer()
                print("\t \t Runtime: ", stop - start)
                outputImageArr = self.saveImage(name, False)
            return outputImageArr

    def initializeClusters(self):

        """Distributes clusters of approximate size S^2 over the image"""

        # (x, y) serves as current coordinates for setting cluster centers
        x = self.S // 2
        y = self.S // 2

        # Run across image and set cluster centers
        while y < self.height:

            while x < self.width:

                # Add new cluster centered at (x, y)
                r , g, b= self.colorArr[y][x]
                cluster = Cluster(x, y, r , g, b)
                self.clusters.append(cluster)

                # Iterate horizontally by the cluster iteration size S
                x += self.S

            # Reset horizontal coordinate, and iterate vertically by S
            x = self.S // 2
            y += self.S

    def updateClusters(self):

        """Execute update if gradient of neighbor is smaller than current gradient"""

        def calculateGradient(x, y):

            """
            Compute the gradient for the pixel with coordinates (x, y) using L2 norm

            Return:
                Gradient from L2 norm of lab-vector

            Input:
                x - (Int) Horizontal Coordinate
                y - (Int) Vertical Coordinate
            """

            # Handle coordinates on edge
            if not (x + 1 < self.width):
                x = self.width - 2

            if not (y + 1 < self.height):
                y = self.height - 2

            # Computes the gradient using L2 norm
            Gx = np.linalg.norm(self.colorArr[y][x + 1] - self.colorArr[y][x - 1], ord=2) ** 2
            Gy = np.linalg.norm(self.colorArr[y + 1][x] - self.colorArr[y - 1][x], ord=2) ** 2
            
            return Gx + Gy


        for cluster in self.clusters:

            currGradient = calculateGradient(cluster.x, cluster.y)

            changeMade = True

            # Continue while gradient is not minimal
            while (changeMade):

                changeMade = False

                # Check gradients on each adjacent pixel and adjust accordingly
                for (dx, dy) in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):

                    _x = cluster.x + dx
                    _y = cluster.y + dy

                    if _x > 0 and _x < self.width - 1 and _y > 0 and _y < self.height - 1:

                        newGradient = calculateGradient(_x, _y)

                        if newGradient < currGradient:

                            changeMade = True

                            _l, _a, _b = self.colorArr[_y][_x]

                            cluster.setCluster(_x, _y, _l, _a, _b)
                            currGradient = newGradient

            cluster.pixelsOfCluster.add((cluster.y, cluster.x))

    def labelPixels(self, labWeight):

        """
            Label each pixel to the closest cluster relative to the LABXY-plane

            Input:
                labWeight - (Float) Value between 0.0 and 1.0 to adjust effectiveness of LAB distance during labeling
        """

        it = self.S * 2

        for cluster in self.clusters:
            
            for y in range(max(cluster.y - it, 0), min(cluster.y + it, self.height)):

                for x in range(max(cluster.x - it, 0), min(cluster.x + it, self.width)):

                    r, g , b = self.colorArr[y][x]

                    labDistance = math.sqrt(( r- cluster.r)**2 + (g - cluster.g )**2 + (b - cluster.b)**2)
                    xyDistance = math.sqrt((x - cluster.x)**2 + (y - cluster.y)**2)

                    # Avoiding scaling xyDistance by 1 / S for prettier superpixels
                    D = labWeight * labDistance + (1 - labWeight) * (self.M) * xyDistance

                    # Update if scaled distance is better than previous minimal distance
                    if D < self.distanceArr[y][x]:

                        pixel = (y, x)

                        # Update label for this pixel
                        if pixel not in self.labels:
                            self.labels[pixel] = cluster
                            cluster.pixelsOfCluster.add(pixel)
                        else:
                            self.labels[pixel].pixelsOfCluster.remove(pixel)
                            self.labels[pixel] = cluster
                            cluster.pixelsOfCluster.add(pixel)

                        self.distanceArr[y][x] = D

    def updateCenters(self):

        """
            Update centers for each clusters by using mean of (x, y) coordinates
        """

        for cluster in self.clusters:

            widthTotal = heightTotal = count = 0

            for p in cluster.pixelsOfCluster:

                heightTotal += p[0]
                widthTotal += p[1]
                count += 1

            if count == 0:
                count = 1

            _x = widthTotal // count
            _y = heightTotal // count
            _r = self.colorArr[_y][_x][0]
            _g = self.colorArr[_y][_x][1]
            _b = self.colorArr[_y][_x][2]

            cluster.setCluster(_x, _y, _r, _g, _b)

    def enforceConnectivity(self):

        """
            Relabels pixels disjoint from cluster center

            First, perform search for all pixels not reachable by cluster center

            Then, relabels each of these pixels to the closest cluster center
        """

    

        def hasValidCoor(p):

            """Return true if pixel boundaries are within image"""

            return (p[0] >= 0) and (p[0] < self.height) and (p[1] >= 0) and (p[1] < self.width) 



        for cluster in self.clusters:
            
            # Execute BFS to find pixels not connected to cluster
            pixelSet = set(cluster.pixelsOfCluster)
            bfsQueue = []
            clusterCenterPixel = (cluster.y, cluster.x)

            # Set s keeps track of pixels not connected to center
            if clusterCenterPixel in pixelSet:
                bfsQueue.append(clusterCenterPixel)
                pixelSet.remove(clusterCenterPixel)
            elif pixelSet:
                bfsQueue.append(next(iter(pixelSet)))

            while bfsQueue:

                pixel = bfsQueue.pop()

                for _p in ((pixel[0] - 1, pixel[1]), (pixel[0] + 1, pixel[1]), (pixel[0], pixel[1] - 1), (pixel[0], pixel[1] + 1)):

                    if hasValidCoor(_p) and (_p in pixelSet):

                        bfsQueue.append(_p)
                        pixelSet.remove(_p)

            # Find new labels for each pixel not connected to cluster
            while pixelSet:

                done = False
                bfsQueue.append(next(iter(pixelSet)))

                while (bfsQueue):

                    # Search for pixel with different label and shortest distance
                    pixel = bfsQueue.pop()

                    for _p in ((pixel[0] - 1, pixel[1]), (pixel[0] + 1, pixel[1]), (pixel[0], pixel[1] - 1), (pixel[0], pixel[1] + 1)):

                        # Different label found, so relabel this pixel using it and move onto relabeling others
                        if hasValidCoor(_p):

                            if _p not in cluster.pixelsOfCluster:

                                # Relabel
                                self.labels[pixel].pixelsOfCluster.remove(pixel)
                                self.labels[pixel] = self.labels[_p]
                                self.labels[_p].pixelsOfCluster.add(pixel)
                                done = True
                                break

                            else:
                                bfsQueue.append(_p)

                    if done:
                        pixelSet.remove(pixel)
                        bfsQueue.clear()

    def saveImage(self, path, isBordered):

        """
            Saves segmented image, along with borders and center indications, into path

            Input:
                path -       (String) File path/name for save location for image 
                isBordered - (Bool)  Boolean value representing if output will have borders around clusters
        """

        def willBeBorder(px, py):

            """
            Return:
                True, if any pixel adjacent to the passed pixel is of a different cluster and not black
                False, otherwise

            Input:
                px - (Int) Horizontal component of pixel
                py - (Int) Vertical component of pixel    
            """

            L = self.labels[(py, px)]

            return (((px == 0) or ((self.labels[(py, px - 1)] != L) )) \
            or ((px == self.width - 1) or ((self.labels[(py, px + 1)] != L)))\
            or ((py == 0) or ((self.labels[(py - 1, px)] != L) )) \
            or ((py == self.height - 1) or (self.labels[(py + 1, px)] != L)))

        
        self.imageArr = np.copy(self.colorArr)

        if isBordered:

            for cluster in self.clusters:

                for p in cluster.pixelsOfCluster:

                    px = p[1]
                    py = p[0]
                    self.imageArr[py][px][0] = cluster.r
                    self.imageArr[py][px][1] = cluster.g
                    self.imageArr[py][px][2] = cluster.b
                    

        else:

            for cluster in self.clusters:

                for p in cluster.pixelsOfCluster:

                    px = p[1]
                    py = p[0]
                    
                    self.imageArr[py][px][0] = cluster.r
                    self.imageArr[py][px][1] = cluster.g
                    self.imageArr[py][px][2] = cluster.b


        io.imsave(path, color.lab2rgb(self.imageArr))

        return color.lab2rgb(self.imageArr)

    
if __name__ == '__main__':


    print("[INFO] SLIC process is started ...")
    process = SLIC('/home/asma/Documents/Programing/BachelorProject/bachelorPro/Bachelor-Project/images/img1.jpg', 5000, 10)
    print("[INFO] important parameters .....")
    outputImageArr = process.execute(1, 0.2, False)
    print("\t \t Original Image width : {} , Height : {} ".format(process.width  , process.height))
    print("\t \t SLIC Values K : {} , M : {} , N : {} , S : {}".format(process.K , process.M , process.N , process.S))
    print("\t \t Number of output image pixels : {} ".format(len(outputImageArr)))
    print("[INFO] superpixel operation is finished ....")
    io.imshow(outputImageArr)

    cv2.imshow("output-+", outputImageArr)
    cv2.waitKey(0)

    with open("/home/asma/Documents/Programing/BachelorProject/bachelorPro/Bachelor-Project/SuperpixeledImage.csv" , 'w') as f :
        writer = csv.writer(f)
        writer.writerow(outputImageArr)
        f.close()
