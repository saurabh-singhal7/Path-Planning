# imports
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# class Graph
class Graph:
    # Initialize the Graph objects with vertices,
    # track image, and adjacency matrix
    def __init__(self, vertices, track):
        self.vertices = vertices
        self.track = track
        self.adj_matrix = np.zeros((len(vertices), len(vertices)))
    
    # add edges between two vertices in the adjacency matrix
    def add_edge(self, v1, v2, weight):
        self.adj_matrix[v1][v2] = weight
        self.adj_matrix[v2][v1] = weight
    
    # calculate euclidean distance between two vertices
    def euclidean_distance(self, v1, v2):
        return euclidean(self.vertices[v1], self.vertices[v2])

    # create the graph
    def create_graph(self):
        # traverse each vertex
        for i in range(len(self.vertices)):
            # combinations with every other vertex
            for j in range(i+1, len(self.vertices)):
                # check if there is a valid edge between two vertices
                if self.is_valid_edge(i, j):
                    # calculate the euclidean distance
                    weight = self.euclidean_distance(i, j)
                    # add the edge in the graph
                    self.add_edge(i, j, weight)
    
    # function to check if an edge is valid
    def is_valid_edge(self, v1, v2):
        # get points in the line from one vertex to the other
        line_x, line_y = self.get_line(self.vertices[v1], self.vertices[v2])
        # traverse points in a line
        for i in range(len(line_x)):
            # check if every point is valid in the line
            if not self.is_valid_point(line_x[i], line_y[i]):
                return False
        return True

    # function to check if a point is valid on the track image
    def is_valid_point(self, x, y):
        # if the BGR values are greater than 205 i.e, it is white area
        # then return true
        if ((self.track[y][x][0] > 205) and (self.track[y][x][1] > 205) and (self.track[y][x][2] > 205)):
            return True
        return False

    # function to return points on a line
    def get_line(self, start, end):
        # get x and y components out of vertices
        start_x, start_y = start[0], start[1]
        end_x, end_y = end[0], end[1]
        # the number of points is the number of whole numbers between
        # points
        points = int(abs(end_x - start_x))
        # store points on a line in an array
        line_x = np.linspace(start_x, end_x, points+1, dtype=int)
        line_y = np.linspace(start_y, end_y, points+1, dtype=int)
        # return the array
        return line_x, line_y
    
    # plot the graph with all the valid edges
    def plot_graph(self):
        # show the track image
        plt.imshow(self.track)
        # traverse every vertex
        for i in range(len(self.vertices)):
            # combinations with rest of the vertices
            for j in range(i+1, len(self.vertices)):
                # if the weight is greater than 0
                # i.e, if there is a valid edge
                if self.adj_matrix[i][j] > 0:
                    # plot the edge
                    x1, y1 = self.vertices[i]
                    x2, y2 = self.vertices[j]
                    plt.plot([x1, x2], [y1, y2], '-g')
        # show the plot
        plt.show()

def selectPoints(image):
    # take 20 points input from the user
    points = plt.ginput(20, 0, True)
    plt.close()
    # return points as an array
    return points

# to indicate if the child does not have a parent
# i.e, no vertex leads up to that child
NO_PARENT = -1

def dijkstra(adjacency_matrix, start_vertex, vertices, track):
    # get the number of vertices
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

    saveSol(adjacency_matrix, parents, vertices, track)

# function to save the solution into a pickle file
# and plot the paths
def saveSol(graph, parents, vertices, track):
    # to store the list of vertices leading upto
    # respective paths
    path1 = []
    path2 = []
    path1 = print_path(1, parents, path1)
    path2 = print_path(2, parents, path2)
    # open the pickle file and dump the paths
    with open('manual_path.pkl', 'wb') as f:
        pickle.dump(graph, f, 3)
        pickle.dump(path1, f, 3)
        pickle.dump(path2, f, 3)
    # plot the paths
    plt.imshow(track)
    path1X = []
    path1Y = []
    path2X = []
    path2Y = []
    for vertex in path1:
        x1, y1 = vertices[vertex]
        path1X.append(x1)
        path1Y.append(y1)
    for vertex in path2:
        x2, y2 = vertices[vertex]
        path2X.append(x2)
        path2Y.append(y2)
    plt.plot(path1X, path1Y, '-g')
    plt.plot(path2X, path2Y, '-r')
    plt.show()

# function to find the path
def print_path(current_vertex, parents, path):
    # Base case : Source node has
    # been processed
    if current_vertex == NO_PARENT:
        return
    # recursive call to explore the path back
    print_path(parents[current_vertex], parents, path)
    # add the vertex to a list and return it
    path.append(current_vertex)
    return path

def main():
    # read in the track image
    demoTrack = cv2.imread("C:/Users/saura/Desktop/Purdue Classes/Spring 2023/MFET 442/Lab 10/knoy_demo_track.jpg")
    # show the image
    plt.imshow(demoTrack)
    # start, goal1, and goal2 points were given
    startPoint = (485, 504)
    goal1 = (60, 277)
    goal2 = (125, 125)
    initialPts = [startPoint, goal1, goal2]
    # get the vertices from the user
    vertices = selectPoints(demoTrack)
    # arrange them into an array
    vertices = np.vstack([initialPts, vertices])
    # make the graph object
    graph = Graph(vertices=vertices, track=demoTrack)
    # create the graph
    graph.create_graph()
    # plot the graph
    graph.plot_graph()
    # find the best path, save it, and plot it
    dijkstra(graph.adj_matrix, 0, vertices, demoTrack)

if __name__ == "__main__":
    main()