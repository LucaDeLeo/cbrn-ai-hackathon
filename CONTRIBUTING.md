# Contributing to RobustCBRN Eval

Thank you for your interest in contributing to RobustCBRN Eval! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. By participating in this project, you agree to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Development Workflow

### Getting Started

1. Fork the repository to your GitHub account
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/robustcbrn-eval.git
   cd robustcbrn-eval
   ```

3. Set up the development environment:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment
   uv venv
   source .venv/bin/activate  # Linux/Mac

   # Install dependencies
   uv pip install -r requirements.txt
   uv pip install -r requirements-dev.txt
   ```

4. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

### Making Changes

1. Make your changes in your feature branch
2. Write or update tests as needed
3. Ensure all tests pass:
   ```bash
   python -m unittest
   ```
4. Run linting and formatting:
   ```bash
   ruff check src/ tests/
   black src/ tests/
   mypy src/
   ```

### Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Format your commit messages as:

```
<type>: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (formatting, missing semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

Examples:
```
feat: add consensus detection module
fix: correct normalization of answer indices
docs: update installation instructions
test: add tests for bootstrap confidence intervals
```

## Pull Request Process

1. Update documentation for any new features or changed behavior
2. Ensure your branch is up to date with the main branch:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request from your fork to the main repository
5. Fill out the pull request template completely
6. Wait for review from maintainers

### PR Review Guidelines

- PRs require at least one approving review before merging
- All CI checks must pass
- Code coverage should not decrease
- Documentation must be updated for new features
- Breaking changes require discussion with maintainers

## Testing Requirements

All contributions must include appropriate tests:

### Unit Tests
- Test individual functions and methods
- Use mocking for external dependencies
- Aim for >70% code coverage
- Place tests in `tests/` directory with `test_` prefix

### Integration Tests
- Test interaction between components
- Verify end-to-end workflows
- Include edge cases and error conditions

### Running Tests
```bash
# Run all tests
python -m unittest

# Run specific test file
python -m unittest tests.test_module

# Run with coverage
coverage run -m unittest
coverage report
coverage html  # Generate HTML report
```

## Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters and returns
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Add docstrings to all public functions and classes

### Documentation Style
- Use clear, concise language
- Include code examples where helpful
- Keep README and documentation up to date
- Document any non-obvious design decisions

## Setting Up Pre-commit Hooks

We recommend using pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files (optional)
pre-commit run --all-files
```

## Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Python version and OS
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Error messages and stack traces
- Minimal reproducible example if possible

### Feature Requests
For feature requests, please describe:
- The problem you're trying to solve
- Your proposed solution
- Alternative solutions you've considered
- Any additional context

## Getting Help

If you need help:
- Check the documentation first
- Search existing issues and discussions
- Ask questions in GitHub Discussions
- Contact maintainers if needed

## Recognition

Contributors will be recognized in:
- AUTHORS.md file
- Release notes
- Project documentation

Thank you for contributing to RobustCBRN Eval!