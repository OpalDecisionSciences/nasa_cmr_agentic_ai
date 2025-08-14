"""
Enhanced Streaming System for NASA CMR Agent.

Provides real-time streaming responses with progress indicators, error handling,
backpressure management, and adaptive performance optimization.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""
    START = "start"
    PROGRESS = "progress"
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    PARTIAL_RESULT = "partial_result"
    ERROR = "error"
    WARNING = "warning"
    COMPLETE = "complete"
    HEARTBEAT = "heartbeat"
    METADATA = "metadata"


class StreamPriority(Enum):
    """Priority levels for stream events."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StreamEvent:
    """Individual stream event."""
    event_type: StreamEventType
    event_id: str
    timestamp: str
    data: Dict[str, Any]
    priority: StreamPriority = StreamPriority.NORMAL
    agent_id: Optional[str] = None
    progress_percent: Optional[float] = None
    estimated_completion: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for streaming."""
        return {
            "event_type": self.event_type.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "priority": self.priority.value,
            "agent_id": self.agent_id,
            "progress_percent": self.progress_percent,
            "estimated_completion": self.estimated_completion
        }
    
    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format."""
        event_data = self.to_dict()
        return f"event: {self.event_type.value}\ndata: {json.dumps(event_data)}\n\n"
    
    def to_ndjson_format(self) -> str:
        """Convert to Newline Delimited JSON format."""
        return json.dumps(self.to_dict()) + "\n"


@dataclass
class StreamMetrics:
    """Streaming performance metrics."""
    stream_id: str
    start_time: float
    events_sent: int = 0
    bytes_sent: int = 0
    errors_count: int = 0
    avg_event_size: float = 0.0
    throughput_events_per_sec: float = 0.0
    client_buffer_size: int = 0
    backpressure_events: int = 0
    
    def update_metrics(self, event_size: int):
        """Update streaming metrics."""
        self.events_sent += 1
        self.bytes_sent += event_size
        
        # Calculate averages
        if self.events_sent > 0:
            self.avg_event_size = self.bytes_sent / self.events_sent
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.throughput_events_per_sec = self.events_sent / elapsed_time


class StreamBuffer:
    """Advanced buffer for managing streaming events."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 50):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.buffer: List[StreamEvent] = []
        self.current_memory = 0
        self.lock = asyncio.Lock()
        
        # Priority queues
        self.priority_buffer = {
            StreamPriority.CRITICAL: [],
            StreamPriority.HIGH: [],
            StreamPriority.NORMAL: [],
            StreamPriority.LOW: []
        }
    
    async def add_event(self, event: StreamEvent) -> bool:
        """Add event to buffer with priority handling."""
        async with self.lock:
            event_size = len(json.dumps(event.to_dict()))
            
            # Check memory limits
            if self.current_memory + event_size > self.max_memory_bytes:
                await self._evict_low_priority_events()
                if self.current_memory + event_size > self.max_memory_bytes:
                    logger.warning("Stream buffer memory limit exceeded")
                    return False
            
            # Check size limits
            if len(self.buffer) >= self.max_size:
                await self._evict_oldest_events()
                if len(self.buffer) >= self.max_size:
                    logger.warning("Stream buffer size limit exceeded")
                    return False
            
            # Add to appropriate priority queue and main buffer
            self.priority_buffer[event.priority].append(event)
            self.buffer.append(event)
            self.current_memory += event_size
            
            return True
    
    async def get_next_events(self, batch_size: int = 10) -> List[StreamEvent]:
        """Get next batch of events prioritized by importance."""
        async with self.lock:
            events = []
            
            # Process by priority order
            for priority in [StreamPriority.CRITICAL, StreamPriority.HIGH, 
                           StreamPriority.NORMAL, StreamPriority.LOW]:
                priority_queue = self.priority_buffer[priority]
                
                while priority_queue and len(events) < batch_size:
                    event = priority_queue.pop(0)
                    events.append(event)
                    
                    # Remove from main buffer
                    if event in self.buffer:
                        self.buffer.remove(event)
                        event_size = len(json.dumps(event.to_dict()))
                        self.current_memory -= event_size
                
                if len(events) >= batch_size:
                    break
            
            return events
    
    async def _evict_low_priority_events(self, count: int = 10):
        """Evict low priority events to free memory."""
        evicted = 0
        for priority in [StreamPriority.LOW, StreamPriority.NORMAL]:
            priority_queue = self.priority_buffer[priority]
            while priority_queue and evicted < count:
                event = priority_queue.pop(0)
                if event in self.buffer:
                    self.buffer.remove(event)
                    event_size = len(json.dumps(event.to_dict()))
                    self.current_memory -= event_size
                    evicted += 1
    
    async def _evict_oldest_events(self, count: int = 10):
        """Evict oldest events to free space."""
        evicted = 0
        while self.buffer and evicted < count:
            event = self.buffer.pop(0)
            
            # Remove from priority queues
            for priority_queue in self.priority_buffer.values():
                if event in priority_queue:
                    priority_queue.remove(event)
            
            event_size = len(json.dumps(event.to_dict()))
            self.current_memory -= event_size
            evicted += 1
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        return {
            "total_events": len(self.buffer),
            "memory_usage_mb": self.current_memory / (1024 * 1024),
            "memory_usage_percent": (self.current_memory / self.max_memory_bytes) * 100,
            "buffer_usage_percent": (len(self.buffer) / self.max_size) * 100,
            "priority_distribution": {
                priority.value: len(queue) 
                for priority, queue in self.priority_buffer.items()
            }
        }


class EnhancedStreamer:
    """Enhanced streaming system with advanced capabilities."""
    
    def __init__(self, stream_id: Optional[str] = None):
        self.stream_id = stream_id or f"stream_{uuid.uuid4().hex[:8]}"
        self.buffer = StreamBuffer()
        self.metrics = StreamMetrics(self.stream_id, time.time())
        self.is_active = False
        self.client_connected = True
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.max_idle_time = 300  # 5 minutes
        self.batch_size = 5
        self.adaptive_batching = True
        
        # Error handling
        self.max_errors = 10
        self.retry_delays = [1, 2, 4, 8, 16]  # exponential backoff
        
        # Callbacks
        self.on_error_callbacks: List[Callable] = []
        self.on_complete_callbacks: List[Callable] = []
        self.on_client_disconnect_callbacks: List[Callable] = []
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_stream(self) -> AsyncGenerator[str, None]:
        """Start streaming with enhanced capabilities."""
        self.is_active = True
        self._start_background_tasks()
        
        # Send stream start event
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.START,
            data={
                "stream_id": self.stream_id,
                "message": "Stream started",
                "capabilities": {
                    "heartbeat": True,
                    "progress": True,
                    "partial_results": True,
                    "error_recovery": True
                }
            },
            priority=StreamPriority.HIGH
        ))
        
        try:
            while self.is_active and self.client_connected:
                # Get next batch of events
                events = await self.buffer.get_next_events(self.batch_size)
                
                if events:
                    for event in events:
                        if not self.client_connected:
                            break
                        
                        try:
                            # Format for streaming
                            formatted_event = event.to_sse_format()
                            
                            # Update metrics
                            self.metrics.update_metrics(len(formatted_event))
                            
                            yield formatted_event
                            
                        except Exception as e:
                            self.metrics.errors_count += 1
                            logger.error(f"Stream event error: {e}")
                            
                            # Send error event
                            await self._emit_event(StreamEvent(
                                event_type=StreamEventType.ERROR,
                                data={"error": str(e), "recoverable": True},
                                priority=StreamPriority.CRITICAL
                            ))
                            
                            if self.metrics.errors_count >= self.max_errors:
                                logger.error("Max errors exceeded, terminating stream")
                                break
                else:
                    # No events, wait a bit
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info(f"Stream {self.stream_id} cancelled")
        except Exception as e:
            logger.error(f"Stream {self.stream_id} error: {e}")
            await self._emit_event(StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e), "fatal": True},
                priority=StreamPriority.CRITICAL
            ))
        finally:
            await self._cleanup_stream()
    
    async def emit_agent_start(self, agent_id: str, agent_name: str, 
                             estimated_duration: Optional[float] = None):
        """Emit agent start event."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.AGENT_START,
            agent_id=agent_id,
            data={
                "agent_name": agent_name,
                "estimated_duration_ms": estimated_duration,
                "status": "starting"
            },
            priority=StreamPriority.HIGH
        ))
    
    async def emit_agent_progress(self, agent_id: str, progress_percent: float,
                                current_step: str, details: Optional[Dict[str, Any]] = None):
        """Emit agent progress event."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.AGENT_PROGRESS,
            agent_id=agent_id,
            progress_percent=progress_percent,
            data={
                "current_step": current_step,
                "details": details or {},
                "status": "in_progress"
            },
            priority=StreamPriority.NORMAL
        ))
    
    async def emit_agent_complete(self, agent_id: str, result: Dict[str, Any],
                                execution_time_ms: float):
        """Emit agent completion event."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.AGENT_COMPLETE,
            agent_id=agent_id,
            progress_percent=100.0,
            data={
                "result": result,
                "execution_time_ms": execution_time_ms,
                "status": "completed"
            },
            priority=StreamPriority.HIGH
        ))
    
    async def emit_agent_error(self, agent_id: str, error: str, 
                             recoverable: bool = True):
        """Emit agent error event."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.AGENT_ERROR,
            agent_id=agent_id,
            data={
                "error": error,
                "recoverable": recoverable,
                "status": "error"
            },
            priority=StreamPriority.CRITICAL
        ))
    
    async def emit_partial_result(self, result_type: str, data: Dict[str, Any],
                                confidence: Optional[float] = None):
        """Emit partial result as it becomes available."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.PARTIAL_RESULT,
            data={
                "result_type": result_type,
                "result_data": data,
                "confidence": confidence,
                "partial": True
            },
            priority=StreamPriority.NORMAL
        ))
    
    async def emit_warning(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Emit warning event."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.WARNING,
            data={
                "message": message,
                "details": details or {}
            },
            priority=StreamPriority.HIGH
        ))
    
    async def emit_metadata(self, metadata: Dict[str, Any]):
        """Emit metadata event."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.METADATA,
            data=metadata,
            priority=StreamPriority.LOW
        ))
    
    async def complete_stream(self, final_result: Optional[Dict[str, Any]] = None):
        """Complete the stream with final result."""
        await self._emit_event(StreamEvent(
            event_type=StreamEventType.COMPLETE,
            data={
                "final_result": final_result,
                "stream_id": self.stream_id,
                "total_events": self.metrics.events_sent,
                "total_bytes": self.metrics.bytes_sent,
                "duration_seconds": time.time() - self.metrics.start_time
            },
            priority=StreamPriority.CRITICAL
        ))
        
        self.is_active = False
        
        # Call completion callbacks
        for callback in self.on_complete_callbacks:
            try:
                await callback(self.stream_id, final_result)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")
    
    async def _emit_event(self, event: StreamEvent):
        """Emit event to stream buffer."""
        if self.is_active:
            success = await self.buffer.add_event(event)
            if not success:
                self.metrics.backpressure_events += 1
                logger.warning("Stream buffer full, event dropped")
    
    async def _heartbeat_loop(self):
        """Send heartbeat events to keep connection alive."""
        while self.is_active and self.client_connected:
            await asyncio.sleep(self.heartbeat_interval)
            
            if self.is_active:
                await self._emit_event(StreamEvent(
                    event_type=StreamEventType.HEARTBEAT,
                    data={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metrics": {
                            "events_sent": self.metrics.events_sent,
                            "throughput": self.metrics.throughput_events_per_sec,
                            "buffer_status": self.buffer.get_buffer_status()
                        }
                    },
                    priority=StreamPriority.LOW
                ))
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _cleanup_stream(self):
        """Clean up stream resources."""
        logger.info(f"Cleaning up stream {self.stream_id}")
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Log final metrics
        duration = time.time() - self.metrics.start_time
        logger.info(
            f"Stream {self.stream_id} completed",
            duration_seconds=duration,
            events_sent=self.metrics.events_sent,
            bytes_sent=self.metrics.bytes_sent,
            errors=self.metrics.errors_count
        )
    
    def add_error_callback(self, callback: Callable):
        """Add error callback."""
        self.on_error_callbacks.append(callback)
    
    def add_complete_callback(self, callback: Callable):
        """Add completion callback."""
        self.on_complete_callbacks.append(callback)
    
    def disconnect_client(self):
        """Mark client as disconnected."""
        self.client_connected = False
        logger.info(f"Client disconnected from stream {self.stream_id}")
        
        # Call disconnect callbacks
        for callback in self.on_client_disconnect_callbacks:
            try:
                asyncio.create_task(callback(self.stream_id))
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get current stream metrics."""
        return {
            "stream_id": self.stream_id,
            "is_active": self.is_active,
            "client_connected": self.client_connected,
            "duration_seconds": time.time() - self.metrics.start_time,
            "events_sent": self.metrics.events_sent,
            "bytes_sent": self.metrics.bytes_sent,
            "errors_count": self.metrics.errors_count,
            "throughput_events_per_sec": self.metrics.throughput_events_per_sec,
            "buffer_status": self.buffer.get_buffer_status()
        }


# Global stream manager for tracking active streams
class StreamManager:
    """Manages multiple active streams."""
    
    def __init__(self):
        self.active_streams: Dict[str, EnhancedStreamer] = {}
        self.stream_metrics: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def create_stream(self, stream_id: Optional[str] = None) -> EnhancedStreamer:
        """Create a new enhanced stream."""
        streamer = EnhancedStreamer(stream_id)
        
        async with self.lock:
            self.active_streams[streamer.stream_id] = streamer
        
        # Add cleanup callback
        async def cleanup_callback(stream_id: str, result: Any):
            async with self.lock:
                if stream_id in self.active_streams:
                    self.stream_metrics[stream_id] = self.active_streams[stream_id].get_stream_metrics()
                    del self.active_streams[stream_id]
        
        streamer.add_complete_callback(cleanup_callback)
        
        logger.info(f"Created stream {streamer.stream_id}")
        return streamer
    
    async def get_stream(self, stream_id: str) -> Optional[EnhancedStreamer]:
        """Get active stream by ID."""
        async with self.lock:
            return self.active_streams.get(stream_id)
    
    async def get_all_streams(self) -> Dict[str, EnhancedStreamer]:
        """Get all active streams."""
        async with self.lock:
            return self.active_streams.copy()
    
    async def disconnect_stream(self, stream_id: str):
        """Disconnect a specific stream."""
        async with self.lock:
            if stream_id in self.active_streams:
                self.active_streams[stream_id].disconnect_client()
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get stream manager statistics."""
        active_count = len(self.active_streams)
        total_events = sum(s.metrics.events_sent for s in self.active_streams.values())
        total_bytes = sum(s.metrics.bytes_sent for s in self.active_streams.values())
        
        return {
            "active_streams": active_count,
            "completed_streams": len(self.stream_metrics),
            "total_events_sent": total_events,
            "total_bytes_sent": total_bytes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global stream manager instance
stream_manager = StreamManager()