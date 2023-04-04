import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class Graph:
    def __init__(self, vertices, track):
        self.vertices = vertices
        self.track = track
        self.adj_matrix = np.zeros((len(vertices), len(vertices)))
    
    def add_edge(self, v1, v2, weight):
        self.adj_matrix[v1][v2] = weight
        self.adj_matrix[v2][v1] = weight
    
    def euclidean_distance(self, v1, v2):
        return euclidean(self.vertices[v1], self.vertices[v2])

    def create_graph(self):
        for i in range(len(self.vertices)):
            for j in range(i+1, len(self.vertices)):
                if self.is_valid_edge(i, j):
                    weight = self.euclidean_distance(i, j)
                    self.add_edge(i, j, weight)
    
    def is_valid_edge(self, v1, v2):
        plt.imshow(self.track)
        line_x, line_y = self.get_line(self.vertices[v1], self.vertices[v2])
        for i in range(len(line_x)):
            if not self.is_valid_point(line_x[i], line_y[i]):
                return False
        return True

    def is_valid_point(self, x, y):
        if ((self.track[y][x][0] > 205) and (self.track[y][x][1] > 205) and (self.track[y][x][2] > 205)):
            return True
        return False

    def get_line(self, start, end):
        start_x, start_y = start[0], start[1]
        end_x, end_y = end[0], end[1]
        points = int(abs(end_x - start_x))
        line_x = np.linspace(start_x, end_x, points+1, dtype=int)
        line_y = np.linspace(start_y, end_y, points+1, dtype=int)
        return line_x, line_y
    
    def plot_graph(self):
        plt.imshow(self.track)
        for i in range(len(self.vertices)):
            for j in range(i+1, len(self.vertices)):
                if self.adj_matrix[i][j] > 0:
                    x1, y1 = self.vertices[i]
                    x2, y2 = self.vertices[j]
                    plt.plot([x1, x2], [y1, y2], '-g')
        plt.show()

def selectPoints(image):
    points = plt.ginput(20, 0, True)
    plt.close()
    return points

def main():
    demoTrack = cv2.imread("C:/Users/saura/Desktop/Purdue Classes/Spring 2023/MFET 442/Lab 10/knoy_demo_track.jpg")
    plt.imshow(demoTrack)
    startPoint = (485, 504)
    goal1 = (60, 277)
    goal2 = (125, 125)
    initialPts = [startPoint, goal1, goal2]
    vertices = selectPoints(demoTrack)
    vertices = np.vstack([initialPts, vertices])
    graph = Graph(vertices=vertices, track=demoTrack)
    graph.create_graph()
    graph.plot_graph()

if __name__ == "__main__":
    main()