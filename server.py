"""
server.py — Entry point for dev-memory-mcp.

Initializes the database, registers all tools, and starts the MCP server
over stdio (for Claude Desktop compatibility).
"""

from mcp.server.fastmcp import FastMCP

import db
from tools import register_tools

# Initialize DB schema on startup (no-op if already exists)
db.init_db()

# Create the MCP server instance
mcp = FastMCP("dev_memory_mcp")

# Register all tools onto the server
register_tools(mcp)

if __name__ == "__main__":
    mcp.run()  # stdio transport — default for Claude Desktop
