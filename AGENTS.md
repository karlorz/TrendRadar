# AGENTS.md: TrendRadar Project Context

This document provides a comprehensive overview of the TrendRadar project, its architecture, and operational procedures for an AI assistant.

## 1. Project Overview

**TrendRadar** is a Python-based application designed to aggregate, analyze, and report on news and trending topics from over 11 different platforms (like Zhihu, Weibo, Douyin, etc.). It helps users filter out noise and focus on personally relevant information.

### Key Features:

*   **Data Aggregation:** Crawls multiple news and social media platforms to collect trending topics.
*   **Content Filtering:** Uses a customizable keyword list (`config/frequency_words.txt`) to filter and group news based on user interests. Supports required words (`+`), normal words, and filter words (`!`).
*   **Personalized Ranking:** Employs a configurable weighting algorithm (`rank_weight`, `frequency_weight`, `hotness_weight` in `config.yaml`) to re-rank all aggregated news according to user preference, rather than platform algorithms.
*   **Multi-Channel Reporting:** Pushes formatted reports to various channels, including WeCom, Feishu, DingTalk, Telegram, Email, and ntfy. It also generates a static HTML report (`index.html`) suitable for viewing on GitHub Pages.
*   **AI-Powered Analysis:** Includes an MCP (Model Context Protocol) server (`mcp_server/`) that exposes a rich set of tools for an AI to perform deep, conversational analysis on the collected news data.
*   **Data Persistence:** Saves crawled data to structured text files in the `output/` directory, organized by date. This historical data is used for trend and frequency analysis.

### Architecture:

The project consists of two main components:

1.  **Crawler & Reporter (`main.py`):** The core engine that runs on a schedule. It fetches data, saves it, analyzes it based on the configuration, and sends out reports.
2.  **MCP Server (`mcp_server/server.py`):** An optional, locally-run server that provides a tool-based API for AI clients (like IDE extensions) to query the data in the `output/` directory.

## 2. Building and Running

There are two primary methods for deploying and running TrendRadar.

### Method 1: GitHub Actions (Fork-based)

This is the simplest, zero-server method.

*   **Setup:**
    1.  The user forks the repository.
    2.  Webhook URLs and other secrets are stored securely in `Settings > Secrets and variables > Actions`.
    3.  User customizes `config/config.yaml` and `config/frequency_words.txt`.
*   **Execution:**
    *   The GitHub Actions workflow defined in `.github/workflows/crawler.yml` runs automatically on a schedule (e.g., hourly).
    *   The workflow installs Python dependencies from `requirements.txt` and executes `main.py`.
    *   The script reads its configuration and uses the secrets injected as environment variables for notifications.
*   **Data Persistence:**
    *   After `main.py` runs, it creates new files in the `output/` directory and updates `index.html`.
    *   The final step of the workflow commits these changes back to the user's forked repository, effectively saving the state and updating the GitHub Pages website.

**Key Commands (for reference, automated by the workflow):**
*   **Install Dependencies:** `pip install -r requirements.txt`
*   **Run Crawler:** `python main.py`
*   **Commit Results:** `git add -A && git commit -m "..." && git push`

### Method 2: Docker (Self-Hosted)

This method provides more control over the execution environment and schedule.

*   **Setup:**
    1.  The user prepares the `config/` directory with their custom settings.
    2.  The user creates a `.env` file to store webhook secrets and the cron schedule.
    3.  The user runs the application using the provided `docker/docker-compose.yml`.
*   **Execution:**
    *   The Docker container uses a pre-built image (`wantcat/trendradar:latest`).
    *   An internal cron-like scheduler inside the container runs the `main.py` script based on the `CRON_SCHEDULE` environment variable.
    *   Environment variables from the `.env` file are used for configuration and secrets, which can override settings in `config.yaml`.
*   **Data Persistence:**
    *   The `config` and `output` directories are mounted as volumes (`-v ../config:/app/config` and `-v ../output:/app/output`). This ensures that configuration and crawled data are stored on the host machine and persist across container restarts.

**Key Commands (for user):**
*   **Start Service:** `docker-compose pull && docker-compose up -d`
*   **View Logs:** `docker logs -f trend-radar`
*   **Manual Run:** `docker exec -it trend-radar python manage.py run`
*   **Stop Service:** `docker-compose down`

## 3. Development Conventions

*   **Configuration Management:** The primary configuration is in `config/config.yaml`. The script is designed to allow environment variables to override any setting, which is the standard practice for secure and flexible deployment in containers and CI/CD environments.
*   **Dependency Management:** Python dependencies are explicitly listed in `requirements.txt` and `pyproject.toml`.
*   **Code Structure:**
    *   `main.py`: A single-file script containing the core crawling and reporting logic. It is structured with classes and functions for different responsibilities (e.g., `DataFetcher`, `PushRecordManager`, report generation, notification sending).
    *   `mcp_server/`: A modular package for the AI analysis server. It separates logic into different tool categories (`data_query.py`, `analytics.py`, etc.) for clarity.
*   **Data Flow:**
    1.  Crawl data from API.
    2.  Save raw, sorted data to `output/{date}/txt/{time}.txt`.
    3.  Read all of today's text files from `output/`.
    4.  Filter and analyze titles based on `frequency_words.txt`.
    5.  Calculate weights and sort news.
    6.  Generate reports (HTML, Feishu markdown, etc.).
    7.  Send notifications.
    8.  Commit updated `output/` and `index.html` to Git (if using Actions).
*   **Testing:** No automated testing framework (like `pytest`) is apparent in the repository structure. Manual testing is likely done via the `workflow_dispatch` trigger in GitHub Actions or by running the script locally.
