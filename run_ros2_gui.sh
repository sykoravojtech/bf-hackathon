#!/bin/bash

# Script to run gui.py with ROS2 Jazzy despite Python version differences
echo "Setting up environment for ROS2 Jazzy with Python 3.10..."

# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# Set Python path to include ROS2 Python libraries
# This allows Python 3.10 to find the Python 3.12 ROS2 packages
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH

# Set ROS_PYTHON_VERSION to ensure compatibility
export ROS_PYTHON_VERSION=3.10

# Print environment information for debugging
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_PYTHON_VERSION: $ROS_PYTHON_VERSION"
echo "PYTHONPATH includes: $(echo $PYTHONPATH | tr ':' '\n' | grep -i ros)"

# Run the modified GUI script with the current Python 3.10 environment
echo "Running GUI script..."
cd /home/vojta/Documents/bf-hackathon
python gui.py

