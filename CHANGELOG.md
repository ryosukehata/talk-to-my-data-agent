# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.1.3] - 2025-02-03

### Changed
- Better cleansing report, showing more information
- Better memory usage, reducing memory footprint by up to 80%
- LLM is set to GPT 4o (4o mini can struggle with code generation)

## [0.1.3] - 2025-01-30

### Added
- Errors are displayed consistently in the app
- Invalid generated code is displayed on error

### Changed
- Additional modules provided to the code execution function
- Improved date parsing
- Default to GPT 4o mini to be compatible with the trial

## [0.1.2] - 2025-01-29

### Changed
- asyncio based frontend
- general clean-up of the interface
- pandas based analysis dataset
- additional tests
- unified renderer for analysis frontend

## [0.1.1] - 2025-01-24

### Added

- Initial functioning version of Pulumi template for data analyst
- Changelog file to keep track of changes in the project.
- pytest for api functions
