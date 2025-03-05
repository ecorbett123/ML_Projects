import numpy as np
import matplotlib.pyplot as plt

# This takes a long time to run (2-3 hrs)

def fy(y, shortest_path, d):
    total = 0
    for i in range(len(y)):
        for j in range(len(y)):
            if i != j:
                total += (np.linalg.norm(np.array(y[i]) - np.array(y[j])) - shortest_path[i][j])**2
    return total

# derivative with respect to a fixed y_i as calculated in previous part
# y is a n x d matrix
def df_dy(y, fixed_y, shortest_path, d):
    total = np.array([0.0]*d)
    for i in range(len(y)):
        norm = np.linalg.norm(np.array(y[fixed_y]) - np.array(y[i]))
        if norm != 0:
            total += (norm - shortest_path[fixed_y][i])*(np.array(y[fixed_y]) - np.array(y[i]))/norm
    return 4*total

# m is the original D x n matrix, d is the dimension to be reduced to
def find_low_dimension_embedding(m, d):
    k = 10 # k nearest neighbors - needs to be large enough to ensure the graph is one large connected component
    learning_rate = 0.005
    num_iterations = 10
    # construct a k-nearest neighbor graph G on m ->
    # a graph where the n nodes correspond to the n datapoints, and edges correspond to the Euclidean distance (dimension D) between the corresponding datapoints
    # n x n adjacency matrix
    num_points = len(m)
    dist_graph = [[0 for column in range(num_points)]
                      for row in range(num_points)]
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance = np.linalg.norm(np.array(m[i]) - np.array(m[j]))
                dist_graph[i][j] = distance
                dist_graph[j][i] = distance
    graph = [[0 for column in range(num_points)]
                      for row in range(num_points)]
    # keep closest k distances, set the rest to 0 -> directed graph
    neighbors = {}
    for i in range(num_points): # took like 30 sec or so
        row_vals_index = sorted(range(len(dist_graph[i])), key=dist_graph[i].__getitem__)
        # first index will be 0 for same row/col index, skip k indices, set the rest to 0
        neighbors_i = row_vals_index[1:k+1]
        for index in neighbors_i:
            graph[i][index] = dist_graph[i][index]
            graph[index][i] = dist_graph[index][i]

    for i in range(num_points):
        neighbors[i] = []
        for j in range(num_points):
            if graph[i][j] != 0:
                neighbors[i].append(j)

    # populate the shortest path matrix
    shortest_path = get_shortest_path(graph, neighbors)
    #shortest_path = get_shortest_path_csv()
    # write the shortest path matrix to file for later so don't need to recompute each run
    # mat = np.matrix(shortest_path)
    # df = pd.DataFrame(data=mat.astype(float))
    # df.to_csv('outfile_hole.csv', sep=' ', header=False, float_format='%.10f', index=False)

    # intialize y vectors for gradient descent
    y_vectors = []
    for i in range(num_points):
        y = [m[i][k] for k in range(d)]
        y_vectors.append(y)

    for j in range(num_iterations):
        #print("f now equals: " + str(fy(y_vectors, shortest_path, d)))
        # gradient descent -> for each feature, calculate gradient and edit that feature
        gradients = []
        for i in range(num_points):
            # compute the gradient
            gradient = df_dy(y_vectors, i, shortest_path, d)
            gradients.append(gradient)

        for i in range(num_points):
            y_vectors[i] -= learning_rate * gradients[i]


    return shortest_path

def get_shortest_path_csv():
    m = np.loadtxt(open("outfile.csv", "rb"), delimiter=" ", skiprows=0)
    return m

def get_shortest_path(graph, neighbors):
    num_points = len(graph)
    all_dist = []
    for i in range(num_points):
        dist = [1e7] * num_points
        dist[i] = 0
        sptSet = [False] * num_points

        for cout in range(num_points):

            u = minDistance(num_points, dist, sptSet)
            sptSet[u] = True

            for v in neighbors[u]:
                if (graph[u][v] > 0 and
                        sptSet[v] == False and
                        dist[v] > dist[u] + graph[u][v]):
                    dist[v] = dist[u] + graph[u][v]
        all_dist.append(dist)
    return all_dist

def minDistance(num_points, dist, sptSet):
    min = 1e7

    for v in range(num_points):
        if dist[v] < min and sptSet[v] == False:
            min = dist[v]
            min_index = v

    return min_index


def show_3d_matrix(m):
    x = [x[0] for x in m]
    y = [x[1] for x in m]
    z = [x[2] for x in m]
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    plt.show()


def show_2d_matrix(m):
    x = [x[0] for x in m]
    y = [x[1] for x in m]
    plt.scatter(x, y)
    plt.show()

def pca_embedding(m, d):
    data = np.array(m)
    # normalize the data
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    cov = np.cov(data, ddof=1, rowvar=False)

    # get eigenvalues and vectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # sort them
    eigens_arg = sorted(range(len(eigenvalues)), key=eigenvalues.__getitem__, reverse=True)
    eigenvectors_sorted = eigenvectors[:, eigens_arg]

    # pick top d values
    reduced_data = np.matmul(data, eigenvectors_sorted[:, :d])
    show_2d_matrix(reduced_data)

if __name__ == "__main__":

    # Q3:
    # read in data file, get matrix in D dimensions (D x n matrix) (3 x 4000)
    m = []
    with open('swiss_roll.txt', 'r') as f:
        for line in f:
            line = line.strip().split(" ")
            line = [float(x) for x in line if x != ""]
            m.append(line)
    result = find_low_dimension_embedding(m, 2)
    show_2d_matrix(result)
    show_3d_matrix(m)
    pca_embedding(m, 2)

    h = []
    with open('swiss_roll_hole.txt', 'r') as f:
        for line in f:
            line = line.strip().split(" ")
            line = [float(x) for x in line if x != ""]
            h.append(line)
    result = find_low_dimension_embedding(h, 2)
    show_2d_matrix(result)
    show_3d_matrix(h)
    pca_embedding(h, 2)