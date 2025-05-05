#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <atomic>

using namespace std;

class Graph {
private:
    int V;
    vector<vector<int>> adj;

public:
    Graph(int vertices) : V(vertices) {
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected
    }

    void parallelBFS(int start) {
        vector<atomic<bool>> visited(V);
        for (int i = 0; i < V; ++i) visited[i] = false;

        queue<int> q;
        q.push(start);
        visited[start] = true;

        cout << "\nParallel BFS starting from vertex " << start << ": ";

        while (!q.empty()) {
            int levelSize = q.size();
            vector<int> currentLevel;

            // Get all nodes of current level
            for (int i = 0; i < levelSize; ++i) {
                int node = q.front(); q.pop();
                cout << node << " ";
                currentLevel.push_back(node);
            }

#pragma omp parallel for
            for (int i = 0; i < currentLevel.size(); ++i) {
                int u = currentLevel[i];
                for (int v : adj[u]) {
                    if (!visited[v].exchange(true)) {
#pragma omp critical
                        q.push(v);
                    }
                }
            }
        }
        cout << endl;
    }

    void parallelDFSUtil(int u, vector<atomic<bool>> &visited) {
        visited[u] = true;
        cout << u << " ";

#pragma omp parallel
        {
#pragma omp single nowait
            for (int v : adj[u]) {
                if (!visited[v]) {
#pragma omp task
                    parallelDFSUtil(v, visited);
                }
            }
        }
    }

    void parallelDFS(int start) {
        vector<atomic<bool>> visited(V);
        for (int i = 0; i < V; ++i) visited[i] = false;

        cout << "\nParallel DFS starting from vertex " << start << ": ";

#pragma omp parallel
        {
#pragma omp single
            parallelDFSUtil(start, visited);
        }

        cout << endl;
    }
};

int main() {
    int vertices, edges;
    cout << "Enter number of vertices: ";
    cin >> vertices;

    cout << "Enter number of edges: ";
    cin >> edges;

    Graph g(vertices);

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < edges; ++i) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int startNode;
    cout << "Enter starting vertex for BFS/DFS: ";
    cin >> startNode;

    g.parallelBFS(startNode);
    g.parallelDFS(startNode);

    return 0;
}
