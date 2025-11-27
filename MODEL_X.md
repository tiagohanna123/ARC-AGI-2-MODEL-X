# ARC-AGI-2 Model X

## equilibriumai v2

Created by Tiago Hanna (github.com/tiagohanna123/o)

A meta-system for solving ARC-AGI-2 tasks using pattern recognition and transformation logic.

## Installation

```bash
# Clone the repository
git clone https://github.com/tiagohanna123/ARC-AGI-2-MODEL-X.git
cd ARC-AGI-2-MODEL-X

# No additional dependencies required (uses Python standard library)
# Python 3.8+ is required
```

## Usage

### Solve All Evaluation Tasks

```bash
python solve.py
```

### Solve a Specific Task

```bash
python solve.py data/evaluation/0934a4d8.json
```

### Run on Sample Tasks

```bash
python solve.py --sample
```

### Generate Submission File

```bash
python solve.py --submit
```

### View System Prompt

```bash
python solve.py --prompt
```

### Run on Training Tasks

```bash
python solve.py --training
```

## Project Structure

```
ARC-AGI-2-MODEL-X/
├── data/
│   ├── evaluation/     # 120 evaluation tasks
│   └── training/       # 1000 training tasks
├── model_x/
│   ├── __init__.py     # Package initialization
│   ├── system_prompt.py  # equilibriumai v2 prompts
│   └── solver.py       # ARC-AGI-2 solver logic
├── solve.py            # Main entry point
├── requirements.txt    # Python dependencies
└── MODEL_X.md          # This file
```

## Model X Features

### Pattern Recognition

The solver implements pattern recognition for:
- Geometric transformations (rotation, flip, transpose)
- Color mapping detection
- Object extraction and segmentation
- Size change analysis
- Local transformation learning

### Transformation Detection

For each ARC-AGI task, the solver:
1. Analyzes training input/output pairs
2. Detects consistent transformation patterns
3. Applies detected patterns to test inputs
4. Generates output predictions

## License

Apache License 2.0 (same as the original ARC-AGI-2 repository)
