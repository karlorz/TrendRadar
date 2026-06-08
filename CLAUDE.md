TrendRadar is a Chinese news hot topic aggregator and analysis tool. It crawls trending news from 11+ platforms (Zhihu, Weibo, Douyin, Baidu, etc.), applies keyword filtering, and pushes notifications to various channels (Feishu, DingTalk, WeChat Work, Telegram, Email, Slack, etc.). It also includes an MCP (Model Context Protocol) server for AI-powered news analysis.

## Common Commands

### Quick Start (Makefile)
```bash
make dev       # Development (skips root index.html, keeps git clean)
make run       # Production (full output including root index.html)
make mcp       # Start MCP server (STDIO)
make mcp-http  # Start MCP server (HTTP on port 3333)
make install   # Install dependencies with uv
make clean     # Remove generated output files
make help      # Show all available commands
```

### Running the main crawler/analyzer
```bash
# Using uv (recommended)
uv run python -m trendradar

# Development mode (skips root index.html to avoid git pollution)
SKIP_ROOT_INDEX=true uv run python -m trendradar

# Using pip
pip install -r requirements.txt
python -m trendradar
```

### Running the MCP server
```bash
# STDIO mode (for MCP clients like Claude Desktop)
uv run python -m mcp_server.server

# HTTP mode (for web-based clients)
uv run python -m mcp_server.server --transport http --port 3333
```

### Docker deployment
```bash
cd docker
docker compose up -d trendradar              # Main crawler service
docker compose up -d trendradar-mcp          # MCP AI analysis service
```

### Manual testing scripts
```bash
# Windows
setup-windows.bat       # Install dependencies
start-http.bat          # Start HTTP MCP server

# Mac/Linux
./setup-mac.sh          # Install dependencies
./start-http.sh         # Start HTTP MCP server
```

## Architecture

### Package Structure

```
trendradar/           # Main application package
├── __main__.py       # Entry point - NewsAnalyzer orchestrates the entire workflow
├── context.py        # AppContext - dependency injection container for the app
├── core/             # Core business logic
│   ├── config.py     # Configuration loading (config.yaml + env vars)
│   ├── analyzer.py   # News analysis and keyword matching
│   ├── frequency.py  # Keyword group parsing and matching
│   └── loader.py     # Data loading utilities
├── crawler/          # Data fetching
│   ├── fetcher.py    # DataFetcher - crawls news from platforms API
│   └── rss/          # RSS feed support
├── storage/          # Data persistence layer
│   ├── manager.py    # StorageManager - facade for storage operations
│   ├── local.py      # LocalStorageBackend (SQLite)
│   └── remote.py     # RemoteStorageBackend (S3-compatible)
├── notification/     # Push notification system
│   ├── dispatcher.py # NotificationDispatcher - routes to all channels
│   ├── senders.py    # Platform-specific senders (Feishu, DingTalk, etc.)
│   └── formatters.py # Message format conversion
├── report/           # Report generation
│   └── html.py       # HTML report generation
└── ai/               # AI integration (LiteLLM-based)
    ├── analyzer.py   # AIAnalyzer - generates analysis reports
    └── translator.py # AI translation support

mcp_server/           # MCP Server package
├── server.py         # FastMCP 2.0 server with 21 tools
├── tools/            # Tool implementations
│   ├── data_query.py    # Basic data retrieval
│   ├── analytics.py     # Advanced analysis (trends, sentiment)
│   ├── search_tools.py  # Search functionality
│   └── storage_sync.py  # Remote storage sync
├── services/         # Shared services
│   ├── data_service.py  # Data access layer
│   └── cache_service.py # Caching
└── utils/            # Utilities
    └── date_parser.py   # Natural language date parsing
```

### Key Design Patterns

1. **AppContext (Dependency Injection)**: `trendradar/context.py` provides centralized access to config, storage, time functions. Most components receive the context rather than creating their own dependencies.

2. **Storage Backend Abstraction**: `StorageManager` supports both local SQLite and remote S3-compatible storage (Cloudflare R2, etc.). Backend is auto-selected based on environment.

3. **Mode-based Report Generation**: Three modes control output behavior:
   - `daily`: All news accumulated today
   - `current`: Only currently-on-chart news
   - `incremental`: Only newly appeared news

4. **LiteLLM Integration**: AI features use LiteLLM for unified access to 100+ AI providers (DeepSeek, OpenAI, Gemini, etc.). Model format: `provider/model_name`.

### Data Flow

1. `NewsAnalyzer.run()` orchestrates: crawl → store → analyze → report → notify
2. Platform data fetched via newsnow API → stored in SQLite (`output/news/{date}.db`)
3. Keywords from `config/frequency_words.txt` filter news for reports
4. Notifications sent to all configured channels simultaneously

## Configuration

- `config/config.yaml` - Main config (platforms, modes, AI settings, notification channels)
- `config/frequency_words.txt` - Keyword groups for filtering (supports regex: `/pattern/`)
- `config/ai_analysis_prompt.txt` - Custom AI analysis prompt template
- `.github/workflows/crawler.yml` - GitHub Actions schedule (cron)

### Environment Variables

Key secrets for GitHub Actions or Docker:
- `AI_API_KEY`, `AI_MODEL` - AI configuration
- `S3_*` - Remote storage (R2/S3)
- Channel-specific: `FEISHU_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `SLACK_WEBHOOK_URL`, etc.

## Testing Notes

- No formal test suite exists; manual testing via `python -m trendradar`
- MCP tools can be tested via MCP Inspector: `npx @modelcontextprotocol/inspector`
- Sample data in `output/` directory for MCP testing

## Code Conventions

- Chinese comments and user-facing messages (target audience is Chinese users)
- Type hints used throughout
- Config values accessed via `ctx.config["KEY"]` pattern
- Async wrappers in MCP server use `asyncio.to_thread()` for sync operations
