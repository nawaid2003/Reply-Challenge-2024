from collections import defaultdict
import heapq
import math

def parse_input_file(file_path):
    """
    Parse the input file and return the necessary data structures.
    """
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
        print(lines)
    width, height, num_golden_points, num_silver_points, num_tile_types = map(int, lines[0].split())

    golden_points = []
    for i in range(num_golden_points):
        x, y = map(int, lines[i + 1].split())
        golden_points.append((x, y))

    silver_points = {}
    for i in range(num_silver_points):
        x, y, score = map(int, lines[i + 1 + num_golden_points].split())
        silver_points[(x, y)] = score

    tile_types = {}
    for i in range(num_tile_types):
        tile_id, cost, count = lines[i + 1 + num_golden_points + num_silver_points].split()
        tile_types[tile_id] = (int(cost), int(count))

    return width, height, golden_points, silver_points, tile_types

def preprocess(width, height, tile_types):
    """
    Create a graph representation of the grid and precompute edge costs.
    """
    graph = defaultdict(dict)
    tile_directions = {
        '3': [(0, 1), (0, -1)],
        '5': [(1, 1), (-1, -1)],
        '6': [(0, -1), (0, 1)],
        '7': [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)],
        '9': [(1, 1), (-1, -1)],
        'A': [(0, -1), (0, 1)],
        'B': [(0, 1), (0, -1), (1, 1), (-1, -1)],
        'C': [(0, -1), (0, 1)],
        'D': [(0, -1), (0, 1), (1, 1), (-1, -1)],
        'E': [(0, -1), (0, 1), (0, -1), (0, 1)],
        'F': [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]
    }

    for tile_id, (tile_cost, tile_count) in tile_types.items():
        for x in range(width):
            for y in range(height):
                for dx, dy in tile_directions[tile_id]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        graph[(x, y)][(nx, ny)] = tile_cost
    print(graph)
    return graph

def find_minimum_cost_paths(graph, golden_points, silver_points):
    """
    Find the minimum cost paths between all pairs of Golden Points while collecting
    the maximum score from the Silver Points along the way.
    """
    paths = {}

    for i in range(len(golden_points)):
        start = golden_points[i]
        distances = dijkstra(graph, start, silver_points)

        for j in range(i + 1, len(golden_points)):
            end = golden_points[j]
            cost = distances[end]
            paths[(start, end)] = cost
            paths[(end, start)] = cost
    print("paths")
    return paths

def dijkstra(graph, start, silver_points):
    """
    Dijkstra's algorithm modified to account for Silver Point scores.
    """
    distances = {start: 0}
    pq = [(0, start)]
    print("dijkstra")
    while pq:
        cost, node = heapq.heappop(pq)
        if cost > distances[node]:
            print("here")
            continue
        for neighbor, edge_cost in graph[node].items():
            print("here2")
            score = silver_points.get(neighbor, 0)
            new_cost = cost + edge_cost - score
            if neighbor not in distances or new_cost < distances.get(neighbor,0):
                print("here3")
                distances[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))
    print("distances")
    return distances

def approximate_tsp(paths, golden_points):
    """
    Use the Christofides algorithm to find an approximate minimum cost cycle
    that visits all Golden Points.
    """
    mst = minimum_spanning_tree(paths, golden_points)
    odd_vertices = [v for v in mst if len(mst[v]) % 2 != 0]
    matching = minimum_weight_perfect_matching(paths, odd_vertices)
    eulerian_circuit = combine_mst_and_matching(mst, matching)
    cycle = shortcut_eulerian_circuit(eulerian_circuit)
    print("cycle")
    return cycle

def minimum_spanning_tree(paths, golden_points):
    """
    Create a minimum spanning tree using Prim's algorithm.
    """
    mst = defaultdict(dict)
    start = golden_points[0]
    visited = set()
    pq = [(0, start, None)]
    while pq:
        cost, node, parent = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if parent is not None:
            mst[node][parent] = cost
            mst[parent][node] = cost
        for neighbor, edge_cost in paths.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(pq, (edge_cost, neighbor, node))
    print("mst")
    return mst

def minimum_weight_perfect_matching(paths, odd_vertices):
    """
    Find the minimum weight perfect matching on the odd-degree vertices.
    """
    graph = defaultdict(dict)
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices[i + 1:], start=i + 1):
            cost = paths.get((u, v), float('inf'))
            graph[u][v] = cost
            graph[v][u] = cost

    matching = {}
    visited = set()
    for u in graph:
        if u not in visited:
            path = dfs_for_matching(graph, u, visited)
            if path is not None:
                for u, v in zip(path, path[1:]):
                    matching[u] = v
                    matching[v] = u
    print("matching")
    return matching

def dfs_for_matching(graph, start, visited):
    """
    Depth-First Search for finding a maximum weight augmenting path.
    """
    path = [start]
    while True:
        u = path[-1]
        visited.add(u)
        neighbors = [v for v in graph[u] if v not in visited]
        if not neighbors:
            if len(path) % 2 == 0:
                return None
            else:
                path.pop()
        else:
            v = max(neighbors, key=lambda x: graph[u][x])
            path.append(v)
            if v in path[:-1]:
                return path[path.index(v):]

def combine_mst_and_matching(mst, matching):
    """
    Combine the minimum spanning tree and the matching to create the Eulerian circuit.
    """
    eulerian_circuit = []
    for u, v in matching.items():
        path = find_path_in_mst(mst, u, v)
        eulerian_circuit.extend(path)
    print("eulerian_circuit")
    return eulerian_circuit

def find_path_in_mst(mst, u, v):
    """
    Find the path between two vertices in the minimum spanning tree.
    """
    path = [u]
    while u != v:
        neighbors = [n for n in mst[u] if n not in path]
        if not neighbors:
            path.pop()
            u = path[-1]
        else:
            u = neighbors[0]
            path.append(u)
    print("path")
    return path

def shortcut_eulerian_circuit(eulerian_circuit):
    """
    Convert the Eulerian circuit to the final cycle by shortcutting repeated vertices.
    """
    cycle = []
    visited = set()
    prev_node = None
    for node in eulerian_circuit:
        if node not in visited:
            visited.add(node)
            if prev_node is not None:
                cycle.append((prev_node, node))
            prev_node = node
    print("cycle")
    return cycle

def output_path(cycle, output_file, tile_types):
    """
    Write the final path to the output file in the required format.
    """
    tile_directions = {
        '3': [(0, 1), (0, -1)],
        '5': [(1, 1), (-1, -1)],
        '6': [(0, -1), (0, 1)],
        '7': [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)],
        '9': [(1, 1), (-1, -1)],
        'A': [(0, -1), (0, 1)],
        'B': [(0, 1), (0, -1), (1, 1), (-1, -1)],
        'C': [(0, -1), (0, 1)],
        'D': [(0, -1), (0, 1), (1, 1), (-1, -1)],
        'E': [(0, -1), (0, 1), (0, -1), (0, 1)],
        'F': [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]
    }
    tiles_used = defaultdict(int)
    with open(output_file, 'w') as file:
        for prev_node, node in cycle:
            dx, dy = node[0] - prev_node[0], node[1] - prev_node[1]
            for tile_id, directions in tile_directions.items():
                if (dx, dy) in directions or (-dx, -dy) in directions:
                    if tiles_used[tile_id] < tile_types[tile_id][1]:
                        file.write(f"{tile_id} {node[0]} {node[1]}\n")
                        tiles_used[tile_id] += 1
                    break

def solve(input_file, output_file):
    width, height, golden_points, silver_points, tile_types = parse_input_file(input_file)
    graph = preprocess(width, height, tile_types)
    paths = find_minimum_cost_paths(graph, golden_points, silver_points)
    cycle = approximate_tsp(paths, golden_points)
    output_path(cycle, output_file, tile_types)

# Example usage
input_file = "input.txt"
output_file = "output.txt"
solve(input_file, output_file)