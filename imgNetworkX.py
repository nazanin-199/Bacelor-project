from superPixel import SLIC
from graphProperties import GraphProperties
from graphrole import RecursiveFeatureExtractor, RoleExtractor

import warnings
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np 
import networkx as nx
import cv2
import math
import matplotlib.pyplot as plt 
import pyvis
from pyvis.network import Network
import collections 
import timeit

class NetworkX ():
    def __init__(self , imagePath  , numSuperPixelExc = 2 , labweight = 0.5 , numSegments = 5000  , compactness = 10 , radius = 15 , graphType = "degree_rank" ):
        self.imagePath = imagePath 
        self.numSuperPixelExc = numSuperPixelExc
        self.labweight = labweight
        self.numSegments = numSegments
        self.compactness = compactness
        self.isBordered = False
        self.radius = radius
        self.Graph = nx.Graph()
        self.graphType = graphType 
        self.histDegInfo = None 

        
    def graphImplementation(self):
        image = SLIC(self.imagePath , 5000, 10)
        imageArr = image.execute(self.numSuperPixelExc, self.labweight  , self.isBordered)
        cv2.imshow("" , imageArr) 
        cv2.waitKey(0)


        """
            Construct a Graph based on Complex NetworkX . 
        """
        for i in range( 0 , len(image.clusters)):
            x, y = image.clusters[i].x , image.clusters[i].y
            r, g, b = image.clusters[i].r , image.clusters[i].b , image.clusters[i].g

            self.Graph.add_node((x, y) , rgb = ( r, g, b))

        # print(self.Graph.nodes(data = True))

        
        def euclideanDistance( x1, y1, x2 , y2 ):
            distance = math.sqrt((x1 - x2)** 2 + (y1 - y2)**2)
            return  distance < self.radius , distance 

        def weight ( distance , r, g, b, r1 , g1 , b1):
            rgb = math.sqrt((r - r1)** 2 + (g - g1)**2 + (b - b1)**2)
            w = distance + rgb 
            return w


        for node in self.Graph.nodes(data = True) :
            x , y = node[0][0] , node[0][1]
            r, g, b = node[1]['rgb'][0] ,  node[1]['rgb'][1] ,  node[1]['rgb'][1]
            
            for no in self.Graph.nodes(data = True):
                x1 , y1 = no[0][0] , no[0][1]
                edgeDistance = euclideanDistance( x, y, x1 ,y1)
                
                if edgeDistance[0] == True:
                    r1, g1, b1 = no[1]['rgb'][0] , no[1]['rgb'][1] , no[1]['rgb'][1]
                    w = weight(edgeDistance[1] ,r, g, b, r1 , g1 , b1)
                    self.Graph.add_edge((x,y) , (x1 ,y1) , weight= w )

        # print(len(self.Graph.edges()))
        # print(self.Graph.edges(data= True))

        # remove selfloops from network 
        loops = list(nx.selfloop_edges(self.Graph))
        self.Graph.remove_edges_from(loops)


    def saveGraph(self):
        pass


    def graphVisualization(self) :
        
        def degreeRankPlot(graph):
            degree_sequence = sorted ((d for n , d in graph.degree()) , reverse=True)
            dmax = max(degree_sequence)
            fig = plt.figure(" Degree of Graph " , figsize=(10 ,10))
            # Create a gridspec for adding subplots of different sizes
            axgrid = fig.add_gridspec(5, 4)
            ax0 = fig.add_subplot()
            # Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
            pos = nx.spring_layout(graph)
            nx.draw_networkx_nodes(graph , pos, ax=ax0, node_size=20)
            nx.draw_networkx_edges(graph ,  pos, ax=ax0, alpha=0.4)
            ax0.set_title("Connected components of G")
            ax0.set_axis_off()

            # ax1 = fig.add_subplot(axgrid[3:, :2])
            # ax1.plot(degree_sequence, "b-", marker="o")
            # ax1.set_title("Degree Rank Plot")
            # ax1.set_ylabel("Degree")
            # ax1.set_xlabel("Rank")

            # ax2 = fig.add_subplot(axgrid[3:, 2:])
            # ax2.bar(*np.unique(degree_sequence, return_counts=True))
            # ax2.set_title("Degree histogram")
            # ax2.set_xlabel("Degree")
            # ax2.set_ylabel("# of Nodes")

            fig.tight_layout()
            plt.show()
        degreeRankPlot(self.Graph)

    
    def graphroleCalculation(self):

        # extract features
        feature_extractor = RecursiveFeatureExtractor(self.Graph)
        features = feature_extractor.extract_features()
        print(f'\n[INFO] features extracted from {feature_extractor.generation_count} recursive generations ....')
        print(features)

        # assign node roles
        role_extractor = RoleExtractor(n_roles=None)
        role_extractor.extract_role_factors(features)
        node_roles = role_extractor.roles
        print('\n[INFO] node role assignments ....')
        pprint(node_roles)
        print('\n[INFO] node role membership by percentage ....')
        print(role_extractor.role_percentage.round(2))


        # build color palette for plotting
        unique_roles = sorted(set(node_roles.values()))
        color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
        # map roles to colors
        role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
        # build list of colors for all nodes in G
        node_colors = [role_colors[node_roles[node]] for node in self.Graph.nodes]


        # plot graph
        plt.figure()
        with warnings.catch_warnings():
            # catch matplotlib deprecation warning
            warnings.simplefilter('ignore')
            nx.draw(
                self.Graph,
                pos=nx.spring_layout(self.Graph, seed=42),
                with_labels=True,
                node_color=node_colors,
            )
        plt.show()
      
    def graphExecute(self):
        print("[INFO] superpixel implementation ....")
        self.graphImplementation()
        print("[INFO] graph implementation ....")
        # print(self.Graph.nodes(data = True))
        print("[Info] number of graph edges :{} ".format(len(self.Graph.edges())))
        print("[INFO] graph visualization ....")
        self.graphVisualization()
        print("[INFO] graph properties ....")
        prop = GraphProperties(self.Graph)
        prop.execute()

        start = timeit.default_timer()
        print("[INFO] feature extraction ....")
        self.graphroleCalculation()
        stop = timeit.default_timer()
        print("[INFO] runtime .... ", stop - start)


if __name__ == '__main__':

    imageNet = NetworkX("/home/asma/Documents/Programing/BachelorProject/bachelorPro/Bachelor-Project/images/img1.jpg" , graphType="degree_rank")
    imageNet.graphExecute()
    
