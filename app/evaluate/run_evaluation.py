"""
Main runner script for BPMN evaluation metrics.
Execute this script to calculate and display all metrics.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.evaluate.bpmn_generation_metrics import main

if __name__ == "__main__":
    print("Starting BPMN Node Generation Evaluation...")
    print("-" * 80)
    main()

