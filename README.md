# ATE
Safe regions of work for power grid

## Problem definition

### Problem statement
Given a grid representing a power grid, the goal is to identify all safe regions where work can be performed without interference from obstacles.

### Useful vocabulary
- **Power Grid(graph)**: A grid of cells representing a power grid. Each node has an input current (I) or a voltage (V). Each branch of the grid has a certain admitance (Y) and a max rate of current (Rate_ij).
- **Safe region**: A region of points which is delimited by the conditions of the rates , imposing on the input currents/ voltages of the nodes. The safe region has the following properties:

- **Rectangle**: When talking about a rectangle, we mean an n-dimensional rectangle which is used to the define a safe region of work for the n nodes.  The coordinate system is defined by the input currents/voltages of the nodes. The reason for using a rectangle is that we want to define independent regions of work for each node, which means each dimension must be independent of the others.
The rectangle is defined by the following properties:

## Grids
We will be using 2 grids for analysing the problem.
### Hexagonal prism grid
For testing the methods of finding safe regions, we will use a hexagonal prism with uniform points distribution. In fact we aren't defining a grid, but a set of points in a 3D space.
### Kyte grid
This is a simple grid comprised of 5 nodes.  Nodes 1, 2 and 3 are connected in a triangle. The node 4 is connected to node 3 and node 5 is connected to node 4. Notably node 5 is the ground node.

## Methods for generating rectangles
We have created 3 methods for generating rectangles.

### Method 1: Using the theoretical boundary conditions

### Method 2: Using 2 feasible points
Creates a rectangle using 2 feasible points. Checks if the rectangle is feasible. If it is not feasible, the rectangle is discarded. If it is feasible, the rectangle is added to the list of rectangles. After a maximum number of iterations, the method stops. The maximum number of iterations is defined by the user. The method returns a list of rectangles.

### Method 3: 

## Evaluating the rectangles
We define the best rectangle as the rectangle which contains the most feasible points. For each rectangle the points within are integrated one by one and a tally is computed.

