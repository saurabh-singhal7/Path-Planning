import sys
import cv2
import pickle
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

NO_PARENT = -1

def dijkstra(adjacency_matrix, start_vertex):
    n_vertices = len(adjacency_matrix[0])
 
    # shortest_distances[i] will hold the
    # shortest distance from start_vertex to i
    shortest_distances = [sys.maxsize] * n_vertices
 
    # added[i] will true if vertex i is
    # included in shortest path tree
    # or shortest distance from start_vertex to
    # i is finalized
    added = [False] * n_vertices
 
    # Initialize all distances as
    # INFINITE and added[] as false
    for vertex_index in range(n_vertices):
        shortest_distances[vertex_index] = sys.maxsize
        added[vertex_index] = False
         
    # Distance of source vertex from
    # itself is always 0
    shortest_distances[start_vertex] = 0
 
    # Parent array to store shortest
    # path tree
    parents = [-1] * n_vertices
 
    # The starting vertex does not
    # have a parent
    parents[start_vertex] = NO_PARENT
 
    # Find shortest path for all
    # vertices
    for i in range(1, n_vertices):
        # Pick the minimum distance vertex
        # from the set of vertices not yet
        # processed. nearest_vertex is
        # always equal to start_vertex in
        # first iteration.
        nearest_vertex = -1
        shortest_distance = sys.maxsize
        for vertex_index in range(n_vertices):
            if not added[vertex_index] and shortest_distances[vertex_index] < shortest_distance:
                nearest_vertex = vertex_index
                shortest_distance = shortest_distances[vertex_index]
 
        # Mark the picked vertex as
        # processed
        added[nearest_vertex] = True
 
        # Update dist value of the
        # adjacent vertices of the
        # picked vertex.
        for vertex_index in range(n_vertices):
            edge_distance = adjacency_matrix[nearest_vertex][vertex_index]
             
            if edge_distance > 0 and shortest_distance + edge_distance < shortest_distances[vertex_index]:
                parents[vertex_index] = nearest_vertex
                shortest_distances[vertex_index] = shortest_distance + edge_distance
 
    saveSol(adjacency_matrix, parents)

def saveSol(graph, parents):
    path1 = []
    path2 = []
    path1 = print_path(1, parents, path1)
    path2 = print_path(2, parents, path2)
    with open('manual path.pkl', 'wb') as f:
        pickle.dump(graph, f, 3)
        pickle.dump(path1, f, 3)
        pickle.dump(path2, f, 3)

def print_path(current_vertex, parents, path):
    # Base case : Source node has
    # been processed
    if current_vertex == NO_PARENT:
        return
    print_path(parents[current_vertex], parents, path)
    path.append(current_vertex)
    return path

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
    dijkstra(graph.adj_matrix, 0)

if __name__ == "__main__":
    main()