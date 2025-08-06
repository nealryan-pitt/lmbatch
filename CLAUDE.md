# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMBatch is a Python CLI tool for batch processing text files through LM Studio's local LLM server. It takes a prompt template and applies it to multiple text files, saving the LLM responses with structured output filenames.

## Development Setup

- Python 3.13+ required
- Dependencies: requests, click, pyyaml, tqdm
- Install with: `pip install -e .`
- Virtual environment recommended (see .gitignore for .venv)

## Running the Application

```bash
# Basic usage
python main.py --prompt promptfiles/analyze.txt --input txtfiles/

# With custom output directory and verbose mode  
python main.py -p prompts/summarize.txt -i document.txt -o results/ -v

# Dry run to preview what would be processed
python main.py --prompt prompt.txt --input txtfiles/ --dry-run

# Full example with all options
python main.py \
  --prompt promptfiles/analyze.txt \
  --input txtfiles/ \
  --output results/ \
  --server http://localhost:1234 \
  --model llama-3.2-8b \
  --temperature 0.7 \
  --max-tokens 2048 \
  --concurrent 3 \
  --verbose
```

## Key Commands

- **Development**: `python main.py --help` - Show all CLI options
- **Dry run**: `python main.py -p prompt.txt -i txtfiles/ --dry-run` - Preview without processing
- **Verbose output**: Add `-v` flag for detailed progress information

## Project Structure

- `main.py`: CLI interface and entry point
- `src/`: Core application modules
  - `client.py`: LM Studio API client with retry logic
  - `processor.py`: Batch processing orchestration
  - `file_manager.py`: File I/O operations and validation
  - `config.py`: Configuration management (YAML + environment variables)
- `config.yaml`: Default configuration file
- `promptfiles/`: Directory for prompt templates
- `txtfiles/`: Directory for input text files
- `output/`: Default output directory for processed files

## Configuration

Configuration can be set via:
1. `config.yaml` file (default settings)
2. Command-line arguments (override config file)
3. Environment variables (override both)

Key environment variables:
- `LMBATCH_SERVER_URL`: LM Studio server URL
- `LMBATCH_MODEL`: Model to use
- `LMBATCH_TEMPERATURE`: Sampling temperature
- `LMBATCH_MAX_TOKENS`: Maximum response tokens

## Output Format

Output files are named: `{prompt_name}.{text_name}.txt`
- Example: `analyze.document1.txt`
- Includes optional metadata headers with processing details
- Automatic filename collision handling (adds numbers: `_001`, `_002`, etc.)