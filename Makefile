# TrendRadar Development Makefile

.PHONY: dev run mcp mcp-http install test clean help

# Default target
.DEFAULT_GOAL := help

# Development run (skips root index.html to keep git clean)
dev:
	SKIP_ROOT_INDEX=true uv run python -m trendradar

# Production run (default behavior, writes root index.html for GitHub Pages)
run:
	uv run python -m trendradar

# MCP server (STDIO mode)
mcp:
	uv run python -m mcp_server.server

# MCP server (HTTP mode)
mcp-http:
	uv run python -m mcp_server.server --transport http --port 3333

# Install dependencies
install:
	uv sync

# Run unit tests
test:
	uv run python -m unittest discover -s tests

# Clean generated files
clean:
	rm -rf output/
	rm -f index.html

# Show help
help:
	@echo "TrendRadar Development Commands"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  dev        Run crawler (dev mode, skips root index.html)"
	@echo "  run        Run crawler (production mode, full output)"
	@echo "  mcp        Start MCP server (STDIO)"
	@echo "  mcp-http   Start MCP server (HTTP on port 3333)"
	@echo "  install    Install dependencies with uv"
	@echo "  test       Run unit tests"
	@echo "  clean      Remove generated output files"
	@echo "  help       Show this help message"
