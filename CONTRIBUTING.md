# Contributing to LocalML finetune

Thank you for your interest in contributing! We welcome contributions from everyone. Here's how to get started.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming community.

## How Can I Contribute?

### Reporting Bugs 🐛

Found a bug? Great! Help us fix it.

**Before creating a bug report, check if it's already been reported** by searching the [issues](../../issues).

When creating a bug report, include:
- **Clear title** - What's broken?
- **Description** - What happens vs. what should happen?
- **Steps to reproduce** - How to make it happen again
- **Environment** - OS, Python version, Docker version
- **Logs** - Error messages or stack traces
- **Screenshots** - If applicable

**Example:**
```markdown
**Title:** User app crashes when uploading large images

**Description:**
When I try to upload an image larger than 50MB, the app crashes with a 500 error.

**Steps to reproduce:**
1. Create a test image > 50MB
2. Open http://localhost:8000
3. Upload the image
4. See error

**Environment:**
- OS: macOS 12.0
- Python: 3.9.10
- Docker: 20.10.12

**Error:**
```
MemoryError: unable to allocate 2.5 GiB for an array
```
```

### Suggesting Enhancements 💡

Have an idea? We'd love to hear it!

When suggesting an enhancement:
- **Clear title** - What's the feature?
- **Description** - Why would this be useful?
- **Example usage** - How would you use it?
- **Why this over alternatives** - Why this approach?

**Example:**
```markdown
**Title:** Add batch image processing endpoint

**Description:**
Users can only upload one image at a time. A batch endpoint would let them upload multiple images in one request.

**Example:**
POST /api/v1/batch/upload
Content-Type: multipart/form-data

file1.jpg, file2.jpg, file3.jpg

**Benefits:**
- Faster for users with many images
- Reduces request overhead
- More efficient database operations
```

### Pull Requests 🔄

Ready to code? Here's how to submit a pull request.

#### Setup

1. **Fork the repository** - Click the fork button on GitHub
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sentinel-cloud-vision.git
   cd sentinel-cloud-vision
   ```
3. **Create a branch**
   ```bash
   git checkout -b fix/issue-123-description
   # or
   git checkout -b feature/new-feature-name
   ```

#### Making Changes

1. **Install dependencies**
   ```bash
   cd sentinel-cloud-vision
   pip install -r requirements.txt
   ```

2. **Make your changes**
   - Keep changes focused (one feature/fix per PR)
   - Follow the code style (see below)
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Run linter
   black --check .
   
   # Run type checker (if available)
   mypy .
   ```

4. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Fix: Handle large file uploads correctly

   - Add size validation before processing
   - Stream uploads instead of loading to memory
   - Return clear error messages for oversized files
   
   Fixes #123"
   ```

5. **Push to your fork**
   ```bash
   git push origin fix/issue-123-description
   ```

6. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Fill in the PR template
   - Describe your changes clearly
   - Link to any related issues

#### PR Guidelines

- **One concern per PR** - Don't mix features and fixes
- **Keep it focused** - Smaller PRs are easier to review
- **Update tests** - Add tests for new code
- **Update docs** - Update documentation if behavior changes
- **Follow the template** - Use the PR template provided
- **Be responsive** - Reply to review comments promptly

## Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use [Black](https://github.com/psf/black) for formatting.

```bash
# Format your code
black .

# Check formatting
black --check .
```

### Naming Conventions

```python
# Classes: PascalCase
class ImageProcessor:
    pass

# Functions/variables: snake_case
def process_image(image_path):
    user_id = 123

# Constants: UPPER_SNAKE_CASE
MAX_IMAGE_SIZE = 104857600  # 100MB

# Private: Leading underscore
def _internal_helper():
    pass
```

### Docstrings

```python
def upload_image(user_id: int, file_path: str) -> ImageAsset:
    """Upload an image for a user.
    
    Args:
        user_id: The user uploading the image
        file_path: Path to the image file
        
    Returns:
        ImageAsset: The created image asset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file isn't a valid image
    """
    pass
```

### Type Hints

```python
# Use type hints for clarity
def process_batch(images: list[str], batch_size: int = 32) -> dict[str, Any]:
    """Process a batch of images."""
    pass
```

### Testing

```python
# Tests should be clear and focused
def test_process_large_image():
    """Should handle images up to 100MB."""
    # Arrange
    large_image = create_test_image(size_mb=50)
    
    # Act
    result = process_image(large_image)
    
    # Assert
    assert result is not None
    assert len(result.detections) > 0
```

## Documentation

### Updating Docs

1. **Changes to behavior** → Update relevant docs
2. **New features** → Add to feature docs
3. **API changes** → Update API reference
4. **Setup changes** → Update SETUP_GUIDE.md

### Writing Style

- **Clear over clever** - Prioritize clarity
- **Examples** - Show how to use new features
- **Link to resources** - Point to relevant docs
- **Test examples** - Make sure code examples work

## Development Workflow

### Local Development

```bash
# Start the full stack
./start-local.sh

# Or specific components
docker-compose up

# In another terminal, run tests
pytest tests/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::test_user_model

# Run with coverage
pytest --cov=sentinel_cloud_vision tests/
```

### Database Changes

If your changes modify the database:

1. **Create a migration**
   ```bash
   # Document your schema changes
   # Add migration file to migrations/
   ```

2. **Test the migration**
   ```bash
   # Test on fresh database
   # Test on existing database
   ```

3. **Document the change** in the PR

## Review Process

When you submit a PR:

1. **Automated checks** run (tests, linting)
2. **Maintainers review** your code
3. **You respond to feedback**
4. **Once approved**, a maintainer merges it

### What Reviewers Look For

- ✅ Code quality and style
- ✅ Tests for new code
- ✅ Documentation updates
- ✅ No breaking changes (unless intended)
- ✅ Clear commit messages
- ✅ No security issues

## Getting Help

- **Questions?** Open a discussion in the issues
- **Stuck?** Ask in PR comments or issues
- **Ideas?** Start a discussion before a big PR
- **Docs unclear?** Let us know!

## Recognition

Contributors are recognized in:
- Pull request comments
- Release notes (for significant contributions)
- Contributor list (coming soon)

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

---

## Thank You! 🙏

We appreciate all contributions, no matter how small. Together we're building something amazing!

**Questions?** Open an issue or reach out to the maintainers.

**Ready to contribute?** Check out [good first issues](../../issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) to get started.
