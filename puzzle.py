import time
import psutil
import os
from collections import deque
import heapq

# Goal state for 15 puzzle
GOAL_STATE = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 0]  # 0 = blank
]

# Initial state
INITIAL_STATE = [
    [1, 2, 0, 4],
    [5, 7, 3, 8],
    [9, 6, 11, 12],
    [13, 10, 14, 15]
]

# Moves: Up, Down, Left, Right
MOVES = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}


# Utility functions
def state_to_tuple(state):
    return tuple([tuple(row) for row in state])


def find_blank(state):
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                return i, j
    return None


def move_blank(state, direction):
    x, y = find_blank(state)
    dx, dy = MOVES[direction]
    new_x, new_y = x + dx, y + dy
    if 0 <= new_x < 4 and 0 <= new_y < 4:
        new_state = [list(row) for row in state]
        new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
        return new_state
    return None


def is_goal(state):
    return state == GOAL_STATE


# Heuristics
def misplaced_tiles(state):
    count = 0
    for i in range(4):
        for j in range(4):
            if state[i][j] != 0 and state[i][j] != GOAL_STATE[i][j]:
                count += 1
    return count


def manhattan_distance(state):
    dist = 0
    for i in range(4):
        for j in range(4):
            val = state[i][j]
            if val != 0:
                goal_x, goal_y = divmod(val - 1, 4)
                dist += abs(goal_x - i) + abs(goal_y - j)
    return dist


# BFS
def bfs(initial_state):
    start_time = time.time()
    visited = set()
    queue = deque([(initial_state, "", 0)])
    visited.add(state_to_tuple(initial_state))
    nodes_expanded = 0

    while queue:
        state, path, depth = queue.popleft()
        nodes_expanded += 1

        if is_goal(state):
            end_time = time.time()
            memory_used = psutil.Process(os.getpid()).memory_info().rss // 1024
            return path, nodes_expanded, (end_time - start_time) * 1000, memory_used

        for move in MOVES:
            new_state = move_blank(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
                visited.add(state_to_tuple(new_state))
                queue.append((new_state, path + move, depth + 1))

    return None, nodes_expanded, None, None


# DFS
def dfs(state, depth=0, max_depth=20, visited=None):
    if visited is None:
        visited = set()

    # Check if goal reached
    if state == GOAL_STATE:
        return [], 1, 0, 0

    # Stop if hit max depth limit
    if depth >= max_depth:
        return None, 0, 0, 0

    visited.add(state_to_tuple(state))

    for move in MOVES:
        new_state = move_blank(state, move)
        if new_state and state_to_tuple(new_state) not in visited:
            result, nodes, t, mem = dfs(new_state, depth + 1, max_depth, visited)
            if result is not None:  # Found solution
                return [move] + result, nodes + 1, t, mem

    return None, 0, 0, 0  # No solution found within limit

# A* Search
def astar(initial_state, heuristic):
    start_time = time.time()
    visited = set()
    pq = []
    h = heuristic(initial_state)
    heapq.heappush(pq, (h, 0, initial_state, ""))  # (f, g, state, path)
    visited.add(state_to_tuple(initial_state))
    nodes_expanded = 0

    while pq:
        f, g, state, path = heapq.heappop(pq)
        nodes_expanded += 1

        if is_goal(state):
            end_time = time.time()
            memory_used = psutil.Process(os.getpid()).memory_info().rss // 1024
            return path, nodes_expanded, (end_time - start_time) * 1000, memory_used

        for move in MOVES:
            new_state = move_blank(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
                visited.add(state_to_tuple(new_state))
                new_g = g + 1
                new_h = heuristic(new_state)
                new_f = new_g + new_h
                heapq.heappush(pq, (new_f, new_g, new_state, path + move))

    return None, nodes_expanded, None, None


# Run tests
if __name__ == "__main__":
    print("BFS:")
    path, nodes, t, mem = bfs(INITIAL_STATE)
    print(f"Moves: {path}")
    print(f"Nodes expanded: {nodes}")
    print(f"Time taken: {t:.2f} ms")
    print(f"Memory used: {mem} KB\n")

    print("DFS:")
    path, nodes, t, mem = dfs(INITIAL_STATE)
    print(f"Moves: {path}")
    print(f"Nodes expanded: {nodes}")
    print(f"Time taken: {t:.2f} ms")
    print(f"Memory used: {mem} KB\n")

    print("A* with Misplaced Tiles:")
    path, nodes, t, mem = astar(INITIAL_STATE, misplaced_tiles)
    print(f"Moves: {path}")
    print(f"Nodes expanded: {nodes}")
    print(f"Time taken: {t:.2f} ms")
    print(f"Memory used: {mem} KB\n")

    print("A* with Manhattan Distance:")
    path, nodes, t, mem = astar(INITIAL_STATE, manhattan_distance)
    print(f"Moves: {path}")
    print(f"Nodes expanded: {nodes}")
    print(f"Time taken: {t:.2f} ms")
    print(f"Memory used: {mem} KB\n")
