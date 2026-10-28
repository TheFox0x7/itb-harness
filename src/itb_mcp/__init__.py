"""MCP server for playing Helltaker with an external agent loop.

The server owns game interaction (capture, input, overlays, recording) but not
the conversation.  Any MCP-capable client -- Mistral Vibe in particular -- can
drive the game through the tools defined in :mod:`itb_mcp.server`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
