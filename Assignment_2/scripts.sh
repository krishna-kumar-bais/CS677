#!/bin/bash

# Install required packages
echo "Installing required packages..."

# Update package index (optional)
echo "Updating package index..."
pip install --upgrade pip

# Install mpi4py
echo "Installing mpi4py..."
pip install mpi4py

# Install numpy
echo "Installing numpy..."
pip install numpy

# Install matplotlib (optional, not used in the provided Python code)
echo "Installing pillow..."
pip install pillow

echo "Installation complete."