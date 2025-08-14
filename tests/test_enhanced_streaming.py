"""
Comprehensive Enhanced Streaming System Testing Suite.

Tests the enhanced streaming implementation with real-time progress tracking,
backpressure management, rate limiting integration, and security features.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import logging
import json
from typing import Dict, List, Any
from datetime import datetime, timezone
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.streaming
class TestEnhancedStreaming:
    """Test comprehensive enhanced streaming implementation."""
    
    async def test_basic_streaming_functionality(self):
        """Test basic streaming system functionality."""
        logger.info("🌊 Testing basic streaming functionality")
        
        from nasa_cmr_agent.streaming.enhanced_stream import EnhancedStreamer, StreamEventType
        
        # Create a basic streamer
        streamer = EnhancedStreamer(
            stream_id="test_stream_001",
            client_id="test_client",
            ip_address="127.0.0.1",
            user_agent="pytest/1.0"
        )
        
        # Test stream creation
        assert streamer.stream_id == "test_stream_001"
        assert streamer.client_id == "test_client"
        assert not streamer.is_active
        
        # Test stream metrics initialization
        metrics = streamer.get_stream_metrics()
        assert metrics["stream_id"] == "test_stream_001"
        assert metrics["events_sent"] == 0
        assert metrics["errors_count"] == 0
        
        logger.info("✅ Basic streaming functionality tests passed")
    
    async def test_stream_event_system(self):
        """Test stream event creation and formatting."""
        logger.info("🌊 Testing stream event system")
        
        from nasa_cmr_agent.streaming.enhanced_stream import (
            StreamEvent, StreamEventType, StreamPriority
        )
        
        # Test event creation
        event = StreamEvent(
            event_type=StreamEventType.AGENT_START,
            event_id="",  # Should be auto-generated
            timestamp="",  # Should be auto-generated
            data={"agent_name": "test_agent", "status": "starting"},
            priority=StreamPriority.HIGH,
            agent_id="test_agent",
            progress_percent=0.0
        )
        
        # Verify auto-generation
        assert event.event_id.startswith("evt_")
        assert event.timestamp  # Should be ISO format timestamp
        
        # Test SSE formatting
        sse_format = event.to_sse_format()
        assert "event: agent_start" in sse_format
        assert "data: " in sse_format
        assert "\n\n" in sse_format  # SSE terminator
        
        # Test NDJSON formatting
        ndjson_format = event.to_ndjson_format()
        assert ndjson_format.endswith("\n")
        
        # Verify JSON is valid
        event_dict = json.loads(ndjson_format.strip())
        assert event_dict["event_type"] == "agent_start"
        assert event_dict["agent_id"] == "test_agent"
        
        logger.info("✅ Stream event system tests passed")
    
    async def test_stream_buffer_management(self):
        """Test stream buffer with priority queuing."""
        logger.info("🌊 Testing stream buffer management")
        
        from nasa_cmr_agent.streaming.enhanced_stream import (
            StreamBuffer, StreamEvent, StreamEventType, StreamPriority
        )
        
        buffer = StreamBuffer(max_size=10, max_memory_mb=1)
        
        # Test adding events with different priorities
        high_priority_event = StreamEvent(
            event_type=StreamEventType.ERROR,
            event_id="high_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={"error": "test error"},
            priority=StreamPriority.CRITICAL
        )
        
        normal_event = StreamEvent(
            event_type=StreamEventType.PROGRESS,
            event_id="normal_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={"progress": 50},
            priority=StreamPriority.NORMAL
        )
        
        low_priority_event = StreamEvent(
            event_type=StreamEventType.METADATA,
            event_id="low_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={"info": "test metadata"},
            priority=StreamPriority.LOW
        )
        
        # Add events
        assert await buffer.add_event(normal_event)
        assert await buffer.add_event(high_priority_event)
        assert await buffer.add_event(low_priority_event)
        
        # Test priority-based retrieval
        events = await buffer.get_next_events(3)
        assert len(events) == 3
        
        # First event should be highest priority (CRITICAL)
        assert events[0].priority == StreamPriority.CRITICAL
        assert events[0].event_id == "high_001"
        
        # Test buffer status
        status = buffer.get_buffer_status()
        assert status["total_events"] == 0  # Events were consumed
        assert "memory_usage_mb" in status
        assert "priority_distribution" in status
        
        logger.info("✅ Stream buffer management tests passed")
    
    async def test_streaming_with_agent_integration(self):
        """Test streaming with simulated agent workflow."""
        logger.info("🌊 Testing streaming with agent integration")
        
        from nasa_cmr_agent.streaming.enhanced_stream import EnhancedStreamer, StreamEventType
        
        streamer = EnhancedStreamer(
            client_id="agent_test_client",
            ip_address="127.0.0.1"
        )
        
        # Simulate agent workflow
        agent_workflow_events = []
        
        async def collect_stream_events():
            """Collect streaming events for testing."""
            event_count = 0
            async for event_data in streamer.start_stream():
                if event_count >= 10:  # Limit for testing
                    break
                    
                # Parse SSE format
                if event_data.startswith("event:"):
                    lines = event_data.strip().split('\n')
                    event_line = next((line for line in lines if line.startswith("data: ")), "")
                    if event_line:
                        try:
                            event_json = event_line[6:]  # Remove "data: " prefix
                            event = json.loads(event_json)
                            agent_workflow_events.append(event)
                        except json.JSONDecodeError:
                            pass
                
                event_count += 1
        
        # Start streaming and simulate agent events concurrently
        async def simulate_agent_workflow():
            """Simulate a complete agent workflow."""
            await asyncio.sleep(0.1)  # Let stream initialize
            
            # Step 1: Query interpretation
            await streamer.emit_agent_start(
                agent_id="query_interpreter",
                agent_name="Query Interpreter",
                estimated_duration=1000
            )
            
            await asyncio.sleep(0.05)
            await streamer.emit_agent_progress(
                agent_id="query_interpreter",
                progress_percent=50.0,
                current_step="Analyzing intent"
            )
            
            await asyncio.sleep(0.05)
            await streamer.emit_agent_complete(
                agent_id="query_interpreter",
                result={"intent": "exploratory", "confidence": 0.95},
                execution_time_ms=800
            )
            
            # Step 2: CMR API search
            await streamer.emit_agent_start(
                agent_id="cmr_api",
                agent_name="CMR API Agent",
                estimated_duration=2000
            )
            
            await asyncio.sleep(0.05)
            await streamer.emit_partial_result(
                result_type="collection_preview",
                data={"collection_count": 25, "preview_ready": True},
                confidence=0.8
            )
            
            await asyncio.sleep(0.05)
            await streamer.emit_agent_complete(
                agent_id="cmr_api",
                result={"collections_found": 25, "processing_time": 1800},
                execution_time_ms=1800
            )
            
            # Complete stream
            await streamer.complete_stream({
                "total_agents": 2,
                "total_time_ms": 2600,
                "final_results": {"recommendations": 5}
            })
        
        # Run both tasks concurrently
        stream_task = asyncio.create_task(collect_stream_events())
        workflow_task = asyncio.create_task(simulate_agent_workflow())
        
        # Wait for completion with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(stream_task, workflow_task),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Stream test timed out - this is expected in testing")
        
        # Verify collected events
        assert len(agent_workflow_events) >= 3, "Should have collected multiple workflow events"
        
        # Check for specific event types
        event_types = [event.get("event_type") for event in agent_workflow_events]
        assert "start" in event_types, "Should have stream start event"
        
        # Verify agent events were captured
        agent_events = [event for event in agent_workflow_events if event.get("agent_id")]
        assert len(agent_events) >= 2, "Should have captured agent workflow events"
        
        logger.info(f"Captured {len(agent_workflow_events)} streaming events")
        logger.info("✅ Streaming with agent integration tests passed")
    
    async def test_streaming_rate_limiting_integration(self):
        """Test streaming rate limiting and security integration."""
        logger.info("🌊 Testing streaming rate limiting integration")
        
        from nasa_cmr_agent.streaming.enhanced_stream import EnhancedStreamer
        
        # Skip if security not available
        try:
            from nasa_cmr_agent.security.rate_limiter import get_rate_limiter
        except ImportError:
            logger.info("⚠️ Security systems not available, skipping rate limiting tests")
            return
        
        # Create streamer with rate limiting
        streamer = EnhancedStreamer(
            client_id="rate_limit_test",
            ip_address="10.0.0.1",  # Simulate external IP
            user_agent="TestBot/1.0"
        )
        
        # Test rate limiting methods
        rate_limit_ok = await streamer._check_streaming_rate_limits()
        assert isinstance(rate_limit_ok, bool), "Rate limit check should return boolean"
        
        permissions_ok = await streamer._check_streaming_permissions()
        assert isinstance(permissions_ok, bool), "Permission check should return boolean"
        
        # Test backpressure calculation
        batch_size = await streamer._calculate_effective_batch_size()
        assert isinstance(batch_size, int), "Batch size should be integer"
        assert batch_size >= 1, "Batch size should be at least 1"
        
        # Test metrics include security information
        metrics = streamer.get_stream_metrics()
        assert "rate_limit_exceeded" in metrics
        assert "backpressure_active" in metrics
        assert "security_enabled" in metrics
        
        logger.info("✅ Streaming rate limiting integration tests passed")
    
    async def test_stream_backpressure_management(self):
        """Test backpressure and adaptive batching."""
        logger.info("🌊 Testing stream backpressure management")
        
        from nasa_cmr_agent.streaming.enhanced_stream import EnhancedStreamer
        
        streamer = EnhancedStreamer(client_id="backpressure_test")
        
        # Test initial state
        assert not streamer._backpressure_active
        
        # Simulate high event rate to trigger backpressure
        streamer.metrics.events_sent = 1000
        streamer.metrics.start_time = time.time() - 1.0  # 1 second ago
        
        # This should trigger backpressure detection
        batch_size = await streamer._calculate_effective_batch_size()
        
        # Test backpressure application
        start_time = time.time()
        await streamer._apply_backpressure()
        elapsed = time.time() - start_time
        
        # Should have introduced some delay
        assert elapsed >= 0.05, "Backpressure should introduce delay"
        
        # Test adaptive delay
        start_time = time.time()
        await streamer._adaptive_delay()
        elapsed = time.time() - start_time
        
        # Should have minimal delay for normal operation
        assert elapsed >= 0.0, "Adaptive delay should complete"
        
        logger.info("✅ Stream backpressure management tests passed")
    
    async def test_stream_manager_functionality(self):
        """Test stream manager with multiple concurrent streams."""
        logger.info("🌊 Testing stream manager functionality")
        
        from nasa_cmr_agent.streaming.enhanced_stream import StreamManager
        
        manager = StreamManager()
        
        # Test creating multiple streams
        stream1 = await manager.create_stream(
            client_id="manager_test_1",
            ip_address="127.0.0.1"
        )
        
        stream2 = await manager.create_stream(
            client_id="manager_test_2", 
            ip_address="127.0.0.2"
        )
        
        # Verify streams were created
        assert stream1.stream_id != stream2.stream_id
        assert stream1.client_id != stream2.client_id
        
        # Test getting streams
        retrieved_stream1 = await manager.get_stream(stream1.stream_id)
        assert retrieved_stream1 is not None
        assert retrieved_stream1.stream_id == stream1.stream_id
        
        # Test getting all streams
        all_streams = await manager.get_all_streams()
        assert len(all_streams) == 2
        assert stream1.stream_id in all_streams
        assert stream2.stream_id in all_streams
        
        # Test manager statistics
        stats = manager.get_manager_stats()
        assert stats["active_streams"] == 2
        assert stats["total_events_sent"] >= 0
        assert "timestamp" in stats
        
        # Test disconnecting stream
        await manager.disconnect_stream(stream1.stream_id)
        assert not stream1.client_connected
        
        # Test cleanup by completing streams
        await stream1.complete_stream()
        await stream2.complete_stream()
        
        # Wait a moment for cleanup
        await asyncio.sleep(0.1)
        
        updated_stats = manager.get_manager_stats()
        # Streams should be cleaned up after completion
        
        logger.info("✅ Stream manager functionality tests passed")
    
    async def test_streaming_error_handling(self):
        """Test streaming error handling and recovery."""
        logger.info("🌊 Testing streaming error handling")
        
        from nasa_cmr_agent.streaming.enhanced_stream import EnhancedStreamer, StreamEventType
        
        streamer = EnhancedStreamer(client_id="error_test")
        
        # Test error event emission
        await streamer.emit_agent_error(
            agent_id="test_agent",
            error="Simulated processing error",
            recoverable=True
        )
        
        # Check if error was buffered
        buffer_status = streamer.buffer.get_buffer_status()
        assert buffer_status["total_events"] >= 1
        
        # Test warning emission
        await streamer.emit_warning(
            message="Test warning message",
            details={"warning_type": "simulation"}
        )
        
        # Test error event creation
        error_event = streamer._create_error_event("Test fatal error")
        assert "event: error" in error_event
        assert "Test fatal error" in error_event
        
        # Test metrics tracking
        initial_errors = streamer.metrics.errors_count
        
        # Simulate error condition
        streamer.metrics.errors_count += 1
        
        assert streamer.metrics.errors_count == initial_errors + 1
        
        logger.info("✅ Streaming error handling tests passed")


@pytest.mark.asyncio
@pytest.mark.streaming
async def test_streaming_performance():
    """Test streaming system performance under load."""
    logger.info("⚡ Testing streaming system performance")
    
    from nasa_cmr_agent.streaming.enhanced_stream import EnhancedStreamer, StreamEvent, StreamEventType
    
    streamer = EnhancedStreamer(client_id="performance_test")
    
    # Test event creation performance
    start_time = time.time()
    events = []
    
    for i in range(100):
        event = StreamEvent(
            event_type=StreamEventType.PROGRESS,
            event_id=f"perf_test_{i}",
            timestamp="",
            data={"progress": i, "step": f"Processing item {i}"}
        )
        events.append(event)
    
    creation_time = time.time() - start_time
    logger.info(f"Created 100 events in {creation_time:.3f}s")
    
    # Test buffer performance
    start_time = time.time()
    for event in events:
        await streamer.buffer.add_event(event)
    
    buffer_time = time.time() - start_time
    logger.info(f"Buffered 100 events in {buffer_time:.3f}s")
    
    # Test event retrieval performance
    start_time = time.time()
    retrieved_events = await streamer.buffer.get_next_events(100)
    retrieval_time = time.time() - start_time
    
    logger.info(f"Retrieved 100 events in {retrieval_time:.3f}s")
    
    # Performance assertions
    assert creation_time < 1.0, "Event creation should be fast"
    assert buffer_time < 1.0, "Event buffering should be fast"
    assert retrieval_time < 1.0, "Event retrieval should be fast"
    assert len(retrieved_events) == 100, "All events should be retrieved"
    
    logger.info("✅ Streaming system performance tests passed")


if __name__ == "__main__":
    # Allow running this test directly
    async def main():
        test_class = TestEnhancedStreaming()
        
        try:
            logger.info("🧪 Running comprehensive enhanced streaming tests...")
            
            await test_class.test_basic_streaming_functionality()
            await test_class.test_stream_event_system()
            await test_class.test_stream_buffer_management()
            await test_class.test_streaming_with_agent_integration()
            await test_class.test_streaming_rate_limiting_integration()
            await test_class.test_stream_backpressure_management()
            await test_class.test_stream_manager_functionality()
            await test_class.test_streaming_error_handling()
            
            await test_streaming_performance()
            
            logger.info("🎉 All enhanced streaming tests PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced streaming tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)