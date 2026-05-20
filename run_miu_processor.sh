#!/bin/bash

# Shell script to run the MIU Processor automation on Linux/WSL

echo ""
echo "============================================================"
echo "  MIU Batch Processor - Linux/WSL Setup"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Install required packages
echo "Installing required packages..."
python3 -m pip install --upgrade pip > /dev/null 2>&1
python3 -m pip install playwright > /dev/null 2>&1
python3 -m playwright install chromium > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Starting MIU Processor..."
echo ""

# Run the automation script
python3 automation_miu_processor.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Script failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Processing Complete!"
echo "============================================================"
echo ""
