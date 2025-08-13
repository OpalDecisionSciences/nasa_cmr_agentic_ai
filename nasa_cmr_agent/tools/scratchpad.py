"""
Scratchpad Tool with Persistent Memory

Provides agents with a personal scratchpad for taking notes, tracking thoughts,
and maintaining context across interactions with persistent storage.
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

import aiofiles
from pydantic import BaseModel, Field
from redis import asyncio as aioredis

from ..core.config import settings

logger = logging.getLogger(__name__)


class NoteType(str, Enum):
    """Types of notes agents can create."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    TODO = "todo"
    DECISION = "decision"
    LEARNING = "learning"
    ERROR = "error"
    SUCCESS = "success"


class ScratchpadEntry(BaseModel):
    """Individual entry in the scratchpad."""
    id: str = Field(..., description="Unique entry ID")
    agent_id: str = Field(..., description="ID of the agent that created this entry")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note_type: NoteType = Field(..., description="Type of note")
    content: str = Field(..., description="Note content")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    references: List[str] = Field(default_factory=list, description="References to other entries")


class AgentScratchpad:
    """
    Personal scratchpad for an individual agent with persistent memory.
    
    Features:
    - Redis-based real-time storage for active sessions
    - JSON file backup for long-term persistence
    - Searchable notes with tags and references
    - Context preservation across interactions
    """
    
    def __init__(self, agent_id: str, storage_dir: str = "data/scratchpads"):
        self.agent_id = agent_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_path = self.storage_dir / f"{agent_id}_scratchpad.json"
        self.redis_key = f"scratchpad:{agent_id}"
        
        self.entries: Dict[str, ScratchpadEntry] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        self._entry_counter = 0
        
    async def initialize(self):
        """Initialize scratchpad with persistent storage."""
        # Load from file if exists
        await self._load_from_file()
        
        # Connect to Redis if available
        try:
            self.redis_client = await aioredis.from_url(
                settings.redis_url,
                db=settings.redis_db,
                decode_responses=True
            )
            await self._sync_with_redis()
        except Exception as e:
            logger.warning(f"Redis connection failed, using file storage only: {e}")
            self.redis_client = None
    
    async def add_note(
        self,
        content: str,
        note_type: NoteType = NoteType.OBSERVATION,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        references: Optional[List[str]] = None
    ) -> str:
        """
        Add a new note to the scratchpad.
        
        Args:
            content: Note content
            note_type: Type of note
            context: Additional context
            tags: Tags for categorization
            references: References to other entries
            
        Returns:
            Entry ID
        """
        self._entry_counter += 1
        entry_id = f"{self.agent_id}_{self._entry_counter:06d}"
        
        entry = ScratchpadEntry(
            id=entry_id,
            agent_id=self.agent_id,
            note_type=note_type,
            content=content,
            context=context or {},
            tags=tags or [],
            references=references or []
        )
        
        self.entries[entry_id] = entry
        
        # Persist to storage
        await self._persist_entry(entry)
        
        logger.debug(f"Agent {self.agent_id} added note: {entry_id}")
        return entry_id
    
    async def search_notes(
        self,
        query: Optional[str] = None,
        note_type: Optional[NoteType] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 10
    ) -> List[ScratchpadEntry]:
        """
        Search notes based on criteria.
        
        Args:
            query: Text to search in content
            note_type: Filter by note type
            tags: Filter by tags (any match)
            since: Filter by timestamp
            limit: Maximum results to return
            
        Returns:
            List of matching entries
        """
        results = []
        
        for entry in self.entries.values():
            # Apply filters
            if note_type and entry.note_type != note_type:
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            if since and entry.timestamp < since:
                continue
            
            if query and query.lower() not in entry.content.lower():
                continue
            
            results.append(entry)
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results[:limit]
    
    async def get_recent_notes(self, count: int = 5) -> List[ScratchpadEntry]:
        """
        Get most recent notes.
        
        Args:
            count: Number of notes to retrieve
            
        Returns:
            List of recent entries
        """
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_entries[:count]
    
    async def get_related_notes(self, entry_id: str, max_depth: int = 2) -> List[ScratchpadEntry]:
        """
        Get notes related to a specific entry through references.
        
        Args:
            entry_id: Entry to find relations for
            max_depth: Maximum reference depth to traverse
            
        Returns:
            List of related entries
        """
        if entry_id not in self.entries:
            return []
        
        related = set()
        to_process = {entry_id}
        processed = set()
        current_depth = 0
        
        while to_process and current_depth < max_depth:
            next_batch = set()
            
            for eid in to_process:
                if eid in processed:
                    continue
                
                processed.add(eid)
                
                if eid in self.entries:
                    entry = self.entries[eid]
                    
                    # Add referenced entries
                    for ref in entry.references:
                        if ref in self.entries and ref != entry_id:
                            related.add(ref)
                            next_batch.add(ref)
                    
                    # Find entries that reference this one
                    for other_entry in self.entries.values():
                        if eid in other_entry.references and other_entry.id != entry_id:
                            related.add(other_entry.id)
                            next_batch.add(other_entry.id)
            
            to_process = next_batch
            current_depth += 1
        
        return [self.entries[eid] for eid in related if eid in self.entries]
    
    async def summarize_session(self) -> Dict[str, Any]:
        """
        Generate a summary of the current session's notes.
        
        Returns:
            Summary dictionary
        """
        if not self.entries:
            return {
                "agent_id": self.agent_id,
                "total_notes": 0,
                "summary": "No notes recorded"
            }
        
        # Count by type
        type_counts = {}
        for entry in self.entries.values():
            type_counts[entry.note_type] = type_counts.get(entry.note_type, 0) + 1
        
        # Get unique tags
        all_tags = set()
        for entry in self.entries.values():
            all_tags.update(entry.tags)
        
        # Find key decisions and learnings
        decisions = [e for e in self.entries.values() if e.note_type == NoteType.DECISION]
        learnings = [e for e in self.entries.values() if e.note_type == NoteType.LEARNING]
        errors = [e for e in self.entries.values() if e.note_type == NoteType.ERROR]
        
        return {
            "agent_id": self.agent_id,
            "total_notes": len(self.entries),
            "note_types": type_counts,
            "unique_tags": list(all_tags),
            "key_decisions": [{"id": d.id, "content": d.content[:100]} for d in decisions[-3:]],
            "key_learnings": [{"id": l.id, "content": l.content[:100]} for l in learnings[-3:]],
            "recent_errors": [{"id": e.id, "content": e.content[:100]} for e in errors[-3:]],
            "session_start": min(e.timestamp for e in self.entries.values()) if self.entries else None,
            "last_update": max(e.timestamp for e in self.entries.values()) if self.entries else None
        }
    
    async def clear_old_notes(self, days: int = 30):
        """
        Clear notes older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        to_remove = [
            entry_id for entry_id, entry in self.entries.items()
            if entry.timestamp < cutoff
        ]
        
        for entry_id in to_remove:
            del self.entries[entry_id]
        
        if to_remove:
            await self._save_to_file()
            logger.info(f"Cleared {len(to_remove)} old notes for agent {self.agent_id}")
    
    async def export_notes(self, format: str = "json") -> str:
        """
        Export all notes in specified format.
        
        Args:
            format: Export format (json, markdown)
            
        Returns:
            Exported content
        """
        if format == "json":
            return json.dumps(
                [entry.dict() for entry in self.entries.values()],
                default=str,
                indent=2
            )
        elif format == "markdown":
            md_lines = [f"# Scratchpad for Agent: {self.agent_id}\n"]
            
            for entry in sorted(self.entries.values(), key=lambda x: x.timestamp):
                md_lines.append(f"\n## {entry.timestamp.isoformat()} - {entry.note_type.value}")
                md_lines.append(f"\n{entry.content}")
                
                if entry.tags:
                    md_lines.append(f"\n**Tags:** {', '.join(entry.tags)}")
                
                if entry.context:
                    md_lines.append(f"\n**Context:** {json.dumps(entry.context, indent=2)}")
            
            return "\n".join(md_lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _persist_entry(self, entry: ScratchpadEntry):
        """Persist entry to storage."""
        # Save to Redis if available
        if self.redis_client:
            try:
                await self.redis_client.hset(
                    self.redis_key,
                    entry.id,
                    entry.model_dump_json()
                )
                await self.redis_client.expire(self.redis_key, 86400 * 7)  # 7 days TTL
            except Exception as e:
                logger.error(f"Failed to persist to Redis: {e}")
        
        # Always save to file
        await self._save_to_file()
    
    async def _load_from_file(self):
        """Load entries from file storage."""
        if not self.file_path.exists():
            return
        
        try:
            async with aiofiles.open(self.file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                
                for entry_data in data:
                    entry = ScratchpadEntry(**entry_data)
                    self.entries[entry.id] = entry
                
                if self.entries:
                    # Update counter based on loaded entries
                    max_id = max(
                        int(eid.split('_')[-1]) 
                        for eid in self.entries.keys()
                        if '_' in eid
                    )
                    self._entry_counter = max_id
                
                logger.info(f"Loaded {len(self.entries)} entries for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to load scratchpad from file: {e}")
    
    async def _save_to_file(self):
        """Save entries to file storage."""
        try:
            data = [entry.model_dump() for entry in self.entries.values()]
            
            async with aiofiles.open(self.file_path, 'w') as f:
                await f.write(json.dumps(data, default=str, indent=2))
            
            logger.debug(f"Saved {len(self.entries)} entries to file for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to save scratchpad to file: {e}")
    
    async def _sync_with_redis(self):
        """Sync entries with Redis cache."""
        if not self.redis_client:
            return
        
        try:
            # Get all entries from Redis
            redis_entries = await self.redis_client.hgetall(self.redis_key)
            
            for entry_id, entry_json in redis_entries.items():
                if entry_id not in self.entries:
                    entry = ScratchpadEntry.parse_raw(entry_json)
                    self.entries[entry_id] = entry
            
            # Push file entries to Redis
            for entry in self.entries.values():
                await self.redis_client.hset(
                    self.redis_key,
                    entry.id,
                    entry.model_dump_json()
                )
            
            logger.debug(f"Synced {len(self.entries)} entries with Redis for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to sync with Redis: {e}")
    
    async def close(self):
        """Clean up resources."""
        await self._save_to_file()
        
        if self.redis_client:
            await self.redis_client.close()


class ScratchpadManager:
    """
    Manager for creating and managing agent scratchpads.
    """
    
    def __init__(self, storage_dir: str = "data/scratchpads"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.scratchpads: Dict[str, AgentScratchpad] = {}
    
    async def get_scratchpad(self, agent_id: str) -> AgentScratchpad:
        """
        Get or create a scratchpad for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            AgentScratchpad instance
        """
        if agent_id not in self.scratchpads:
            scratchpad = AgentScratchpad(agent_id, str(self.storage_dir))
            await scratchpad.initialize()
            self.scratchpads[agent_id] = scratchpad
        
        return self.scratchpads[agent_id]
    
    async def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summaries for all active scratchpads.
        
        Returns:
            Dictionary of agent summaries
        """
        summaries = {}
        
        for agent_id, scratchpad in self.scratchpads.items():
            summaries[agent_id] = await scratchpad.summarize_session()
        
        return summaries
    
    async def cleanup_old_data(self, days: int = 30):
        """
        Clean up old data from all scratchpads.
        
        Args:
            days: Number of days to keep
        """
        for scratchpad in self.scratchpads.values():
            await scratchpad.clear_old_notes(days)
    
    async def close_all(self):
        """Close all scratchpads."""
        for scratchpad in self.scratchpads.values():
            await scratchpad.close()


# Global scratchpad manager instance
scratchpad_manager = ScratchpadManager()


from datetime import timedelta