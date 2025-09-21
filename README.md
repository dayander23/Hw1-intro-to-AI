This project solves the 15-Puzzle (sliding puzzle) using different search strategies:
Breadth-First Search (BFS)
Depth-First Search (DFS)
Informed Search (A*) with:
h1 = Number of misplaced tiles
h2 = Manhattan distance
The goal is to move the blank space (0) until the board matches the solved configuration.

How to Run
Make sure you have Python 3 installed.
Install the only extra library we used (for measuring memory):
pip install psutil
Save the code in a file called puzzle.py.
Run the program:
python puzzle.py
