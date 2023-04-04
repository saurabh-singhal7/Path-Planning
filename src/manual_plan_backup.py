import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from PIL import Image

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
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
        line_x, line_y = self.get_line(self.vertices[v1], self.vertices[v2])
        for i in range(len(line_x)):
            if not self.is_valid_point(line_x[i], line_y[i]):
                return False
        return True

    def is_valid_point(self, x, y):
        if x < 0 or x >= self.adj_matrix.shape[0]:
            return False
        if y < 0 or y >= self.adj_matrix.shape[1]:
            return False
        return self.adj_matrix[x][y] == 0

    def get_line(self, start, end):
        start_x, start_y = start
        end_x, end_y = end
        points = abs(end_x - start_x)
        line_x = np.linspace(start_x, end_x, points+1, dtype=int)
        line_y = np.linspace(start_y, end_y, points+1, dtype=int)
        return line_x, line_y

def load_map(map_file):
    img = Image.open(map_file)
    return np.array(img)

def get_points(img, num_points):
    plt.imshow(img)
    points = plt.ginput(num_points, timeout=0, show_clicks=True)
    plt.close()
    return np.array(points, dtype=int)

def plot_graph(graph, img_file):
    img = Image.open(img_file)
    plt.imshow(img)
    for i in range(len(graph.vertices)):
        for j in range(i+1, len(graph.vertices)):
            if graph.adj_matrix[i][j] > 0:
                x1, y1 = graph.vertices[i]
                x2, y2 = graph.vertices[j]
                plt.plot([x1, x2], [y1, y2], color='green')
    plt.show()

def main():
    img_file = 'C:/Users/saura/Desktop/Purdue Classes/Spring 2023/MFET 442/Lab 10/knoy_demo_track.jpg'
    num_points = 23
    start = (502, 484)
    goal1 = (280, 75)
    goal2 = (130, 130)
    img = load_map(img_file)
    points = get_points(img, num_points-3)
    vertices = np.vstack([start, goal1, goal2, points])
    graph = Graph(vertices)
    graph.create_graph()
    plot_graph(graph, img_file)

if __name__ == '__main__':
    main()