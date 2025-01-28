from collections import defaultdict
import time

# This class represents a directed graph using adjacency matrix representation
class Graph:
    def __init__(self, graph):
        self.graph = graph  # Residual graph
        self.ROW = len(graph)

    '''Returns true if there is a path from source 's' to sink 't' in residual graph. 
    Also fills parent[] to store the path'''
    def BFS(self, s, t, parent):
        # Mark all the vertices as not visited
        visited = [False] * self.ROW

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:
            # Dequeue a vertex from queue
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            for ind, val in enumerate(self.graph[u]):
                if not visited[ind] and val > 0:
                    # If we find a connection to the sink node, return true
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True

        # We didn't reach sink in BFS starting from source
        return False

    # Returns the maximum flow from s to t in the given graph
    def FordFulkerson(self, source, sink):
        # This array is filled by BFS and stores the path
        parent = [-1] * self.ROW
        max_flow = 0  # There is no flow initially

        # Augment the flow while there is a path from source to sink
        while self.BFS(source, sink, parent):
            # Find minimum residual capacity of the edges along the path
            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # Update residual capacities of the edges and reverse edges along the path
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow


# Driver code
if __name__ == "__main__":
    # Create the graph given in the above example
    graph = [
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0],
    ]

    g = Graph(graph)
    source = 0
    sink = 5

    # Measure execution time
    start_time = time.time()

    max_flow = g.FordFulkerson(source, sink)

    end_time = time.time()

    # Print the result
    print(f"The maximum possible flow is {max_flow}")
    print(f"Execution time: {end_time - start_time:.6f} seconds")
