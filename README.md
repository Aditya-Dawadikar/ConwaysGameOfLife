# Conway's Game of Life (GPU-Accelerated)

This project is a high-performance simulation of **Conway's Game of Life**, accelerated using **CUDA on the GPU**. The simulation is rendered in real time using **SDL2**, with interactive controls for zooming and panning.

---

## 🧠 What is Conway's Game of Life?

Conway's Game of Life is a **cellular automaton** devised by mathematician **John Conway**. It simulates a grid of cells, each of which can be in one of two states:

- **Alive** (1)
- **Dead** (0)

The simulation progresses in discrete time steps. At each step, the next state of each cell is determined by its neighbors.

---

## 📜 Rules

For each cell in the grid:

1. **Underpopulation**: A live cell with fewer than 2 live neighbors dies.
2. **Survival**: A live cell with 2 or 3 live neighbors lives on.
3. **Overpopulation**: A live cell with more than 3 live neighbors dies.
4. **Reproduction**: A dead cell with exactly 3 live neighbors becomes alive.

These simple rules lead to surprisingly complex emergent behavior.

---

## 🚀 Why GPU?

Simulating a large grid (e.g., 512x512 or higher) on the CPU becomes expensive. The **GPU is ideal** for this task because:
- Each cell's update can be processed in **parallel**
- CUDA threads handle the computation in **parallel blocks**
- **Memory throughput** is optimized for grid-like access patterns

---

## 🖥 Features

- ✅ GPU-accelerated simulation with CUDA
- ✅ Real-time rendering using SDL2
- ✅ Zoom in/out with mouse wheel
- ✅ Pan view by dragging
- ✅ Deterministic pattern spawning (e.g., glider, blinker, glider gun)
- ✅ Two modes: random simulation or single-pattern observation

---

## 🧪 Modes

You can run the app in one of two modes:

### 1. **simulate** (default)
- Initializes a random grid
- Observes emergent patterns

### 2. **observe [pattern]**
- Loads one deterministic pattern at a specific location
- Supported patterns:
  - `glider`
  - `blinker`
  - `block`
  - `toad`
  - `gun`

Example:

```bash
./game_of_life simulate


```bash
./game_of_life observe glider
