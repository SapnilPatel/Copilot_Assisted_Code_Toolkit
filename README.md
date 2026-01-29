# Copilot-Assisted Code Toolkit

A lightweight developer productivity toolkit that streamlines code refactoring,
unit test generation, and technical documentation using repeatable workflows.

This project is designed to reduce repetitive engineering work by combining:
- retrieval of local code context
- standardized templates
- IDE-assisted refactoring using GitHub Copilot 
- basic QA via tests and validation checks

## Features
- Generate refactor patches for legacy Python modules (LLM-pluggable)
- Auto-generate pytest unit tests (LLM-pluggable)
- Produce standardized Markdown documentation (LLM-pluggable + fallback template)
- Simple retrieval of relevant repo context
- Basic QA checks (syntax + pytest)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
