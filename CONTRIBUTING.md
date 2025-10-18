# Contributing to Hybrid Efficient nnU-Net

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Issue Reporting](#issue-reporting)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other contributors

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information
- Other unethical or unprofessional conduct

---

## How to Contribute

### Ways to Contribute

1. **Bug Reports**: Report issues you encounter
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit bug fixes or new features
4. **Documentation**: Improve or expand documentation
5. **Testing**: Add or improve test coverage
6. **Examples**: Create tutorials or example notebooks

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hybrid-nnunet.git
cd hybrid-nnunet
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate
```

### 3. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

**requirements-dev.txt** should include:

```
# Production dependencies
-r requirements.txt

# Development dependencies
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0
mypy>=0.950
pre-commit>=2.17.0
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

### 5. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line Length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized with `isort`
- **Formatting**: Automatic with `black`

### Code Formatting

```bash
# Format code with black
black .

# Sort imports
isort .

# Check with flake8
flake8 .

# Type checking
mypy .
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `HybridEfficientnnUNet`)
- **Functions/Methods**: `snake_case` (e.g., `create_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_EPOCHS`)
- **Private**: Prefix with `_` (e.g., `_internal_method`)

### Documentation

#### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short description of function.

    Longer description if needed, explaining behavior,
    assumptions, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative

    Example:
        >>> result = function_name(5, "test")
        >>> print(result)
        True
    """
    pass
```

#### Type Hints

Always include type hints:

```python
from typing import List, Dict, Optional, Tuple

def process_data(
    data: List[torch.Tensor],
    config: Optional[Dict[str, int]] = None
) -> Tuple[torch.Tensor, float]:
    """Process input data."""
    pass
```

### Code Organization

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path

# 2. Third-party imports
import torch
import torch.nn as nn
import numpy as np

# 3. Local imports
from model import HybridEfficientnnUNet
from config import Config
```

---

## Testing Guidelines

### Writing Tests

Place tests in the `tests/` directory:

```
tests/
├── __init__.py
├── test_model.py
├── test_losses.py
├── test_training.py
└── test_utils.py
```

### Test Structure

```python
import pytest
import torch
from model import HybridEfficientnnUNet

class TestHybridEfficientnnUNet:
    """Test suite for HybridEfficientnnUNet model."""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return HybridEfficientnnUNet(
            in_channels=4,
            num_classes=3,
            base_features=16  # Small for fast testing
        )

    def test_forward_pass(self, model):
        """Test forward pass with valid input."""
        x = torch.randn(1, 4, 64, 64, 64)
        output = model(x)
        assert output.shape == (1, 3, 64, 64, 64)

    def test_invalid_input_shape(self, model):
        """Test error handling for invalid input."""
        x = torch.randn(1, 3, 64, 64, 64)  # Wrong channels
        with pytest.raises(RuntimeError):
            model(x)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestHybridEfficientnnUNet::test_forward_pass

# Verbose output
pytest -v

# Stop at first failure
pytest -x
```

### Test Coverage

Aim for **80%+ code coverage**:

```bash
pytest --cov=. --cov-report=term-missing
```

---

## Pull Request Process

### Before Submitting

1. **Update from main branch**:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:

   ```bash
   black .
   isort .
   flake8 .
   mypy .
   pytest
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

### PR Guidelines

1. **Title**: Clear, descriptive title

   - ✅ "Add Swin Transformer support to encoder"
   - ❌ "Update model.py"

2. **Description**: Include:

   - What changes were made
   - Why the changes were necessary
   - How to test the changes
   - Related issues (if any)

3. **Template**:

   ```markdown
   ## Description

   Brief description of changes

   ## Motivation

   Why is this change needed?

   ## Changes

   - Change 1
   - Change 2

   ## Testing

   How to test these changes

   ## Checklist

   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] All tests passing
   ```

### Review Process

1. At least **1 reviewer approval** required
2. All **CI checks must pass**
3. No **merge conflicts**
4. Up-to-date with **main branch**

### After Approval

```bash
# Squash commits if needed
git rebase -i HEAD~n

# Push to your fork
git push origin feature/your-feature-name
```

---

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
## Bug Description

Clear description of the bug

## To Reproduce

Steps to reproduce:

1. Go to '...'
2. Run '...'
3. See error

## Expected Behavior

What you expected to happen

## Actual Behavior

What actually happened

## Environment

- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 11.7]

## Error Messages
```

Paste error messages here

```

## Additional Context
Any other relevant information
```

### Feature Requests

```markdown
## Feature Description

Clear description of the feature

## Motivation

Why is this feature useful?

## Proposed Solution

How should this be implemented?

## Alternatives Considered

Other approaches you've thought about

## Additional Context

Any other relevant information
```

---

## Development Workflow

### Typical Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-loss-function

# 2. Make changes
# Edit files...

# 3. Test locally
pytest

# 4. Format code
black .
isort .

# 5. Commit changes
git add .
git commit -m "Add Boundary Loss function"

# 6. Push to fork
git push origin feature/new-loss-function

# 7. Create Pull Request on GitHub
```

### Commit Message Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

**Examples:**

```
feat: add Swin Transformer encoder option
fix: resolve memory leak in training loop
docs: update API documentation for losses
test: add unit tests for model components
refactor: simplify attention mechanism code
```

---

## Code Review Checklist

### For Reviewers

- [ ] Code follows project style guidelines
- [ ] Changes are well-documented
- [ ] Tests cover new functionality
- [ ] No unnecessary dependencies added
- [ ] Performance implications considered
- [ ] Backward compatibility maintained
- [ ] Security considerations addressed

### For Contributors

Before requesting review:

- [ ] Self-review completed
- [ ] All tests passing locally
- [ ] Code formatted with black/isort
- [ ] Type hints added
- [ ] Docstrings written
- [ ] Documentation updated
- [ ] No debug code or print statements

---

## Getting Help

- **Documentation**: Read the docs first
- **GitHub Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

---

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in academic citations (if applicable)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Hybrid Efficient nnU-Net! 🎉
