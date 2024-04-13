import numpy as np
import networkx as nx 
import collections 
import matplotlib.pyplot as plt
import math 




class GraphProperties ():
    def __init__(self, imgGraph):
        self.imgGraph = imgGraph
        # information of Histgram degree as zip (deg , cnt)
        degreeSequence = sorted([d for n , d in self.imgGraph] , reverse=True) # n is the node and d is the degree of that node and this is the array of degrees 
        degreeCount = collections.Counter(degreeSequence)
        self.deg , self.cnt = zip(*degreeCount.items())
        print(self.deg , self.cnt)
        self.probability= None
    
    def histogramDegree(self):
        
        fig, ax = plt.subplots()
        plt.bar(self.deg, self.cnt, width=0.8, color='b')
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        plt.show()


    def probabilityDensity(self):
        probability = []
        sum_deg = sum(list(self.cnt))
        for i in range(len(self.deg)):
            pi = self.cnt[i] / sum_deg
            probability.append(pi)
        probability= tuple(probability)
        self.probability = probability

        fig, ax = plt.subplots()
        plt.bar(self.deg,probability, width=0.8, color='b')
        plt.title("Degree Histogram")
        plt.ylabel("Probability")
        plt.xlabel("Degree")
        plt.show()

    def mean_contrast_energy_entropy(self):
        mean =0 
        contrast = 0 
        energy = 0 
        entropy = 0 
        for i in range(len(self.deg)):
            mean += int(self.deg[i]) * float(self.probability[i])
            contrast += float(self.probability[i])* int(self.deg[i])
            energy += float(self.probability[i])**2
            entropy += float(self.probability[i])* math.log(float(self.probability[i]) , 2)
        entropy = -1 *entropy
        
        print("\t \t  Mean : {} , Contrast : {} , Energy : {} , Entropy : {} ".format(mean , contrast , energy , -1*entropy))

    def execute (self):
        print("[INFO] the degree histogram ...")
        self.histogramDegree()
        print("[INFO] probability density histogram ...")
        self.probabilityDensity()
        print("[INFO] degree histogram based information ... ")
        self.mean_contrast_energy_entropy()
