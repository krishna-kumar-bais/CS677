# MPI-Based Volume Rendering

This project implements a parallelized volume rendering algorithm using MPI (Message Passing Interface) and Python. The program divides a large 3D dataset into sub-volumes, distributes these sub-volumes across multiple processes (potentially on multiple machines), and applies ray-casting techniques to generate a final rendered image.

## Features

- **Parallel Processing:** Uses MPI to distribute the computational load across multiple processes, enabling efficient rendering of large datasets.
- **Volume Rendering:** Implements a ray-casting algorithm to render a 3D volume using customizable transfer functions for color and opacity.
- **Customizable Transfer Functions:** Allows users to specify opacity and color transfer functions to control the rendering's appearance.
- **Distributed Computation:** Supports distributed computing across multiple machines with MPI support.

## Requirements

### Software 
    Ensure you have the following software installed:
- **Python 3.x**: The programming language used.
- **MPI4Py**: Python bindings for MPI. Install using `pip install mpi4py`.
- **NumPy**: Required for array manipulations. Install using `pip install numpy`.
- **Pillow (PIL)**: For saving the final rendered image. Install using `pip install pillow`.
- **MPI Implementation**: You need an MPI implementation like [MPICH](https://www.mpich.org/) or [OpenMPI](https://www.open-mpi.org/) for the parallel execution of the program.

### Installation

Install the required Python packages:
`You can install the required packages using pip:`
```bash
pip install mpi4py numpy pillow



 ### Instructions to Run the Volume Rendering Code

Follow these steps to execute the MPI-based volume rendering code:

1. **Ensure Python 3 is Installed**  

2. **Install the Required Dependencies**  

3. **Download and Save the Python Script**  
   Save the provided Python script (e.g., `code.py`) to a directory of your choice.

4. **Prepare the Input Files**  
   - Raw dataset (e.g., `Isabel_1000x1000x200_float32.raw`)
   - Color transfer function (e.g., `color_TF.txt`)
   - Opacity transfer function (e.g., `opacity_TF.txt`)

4. **Command To Run**
 - mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np P --oversubscribe python3 code.py  Isabel_1000x1000x200_float32.raw X Y Z STEP_SIZE opacity_TF.txt color_TF.txt
    - Eg. mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 8 --oversubscribe python3 code.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt 