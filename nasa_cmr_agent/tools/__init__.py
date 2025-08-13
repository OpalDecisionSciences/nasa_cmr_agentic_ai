"""
Tools module for NASA CMR AI Agent

Provides utility tools for agents including scratchpad and other helpers.
"""

from .scratchpad import (
    AgentScratchpad,
    ScratchpadManager,
    ScratchpadEntry,
    NoteType,
    scratchpad_manager
)

__all__ = [
    'AgentScratchpad',
    'ScratchpadManager',
    'ScratchpadEntry',
    'NoteType',
    'scratchpad_manager'
]