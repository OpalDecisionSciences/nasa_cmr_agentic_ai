"""
Advanced Monitoring and Alerting System.

Provides comprehensive system monitoring, intelligent alerting, and
real-time observability for the NASA CMR Agent system with
customizable dashboards and automated incident response.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MonitoringScope(Enum):
    """Monitoring scope levels."""
    SYSTEM = "system"
    SERVICE = "service"
    COMPONENT = "component"
    ENDPOINT = "endpoint"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: Union[float, int]
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class Metric:
    """Metric definition and data."""
    name: str
    type: MetricType
    description: str
    unit: str
    labels: Set[str]
    data_points: deque
    
    def __post_init__(self):
        if not hasattr(self, 'data_points'):
            self.data_points = deque(maxlen=10000)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 100", "== 0"
    severity: AlertSeverity
    duration_seconds: int = 60  # How long condition must be true
    cooldown_seconds: int = 300  # Minimum time between alerts
    labels: Dict[str, str] = None
    notification_channels: List[str] = None
    auto_resolve: bool = True
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.notification_channels is None:
            self.notification_channels = []


@dataclass
class Alert:
    """Active alert."""
    alert_id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    triggered_at: str
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    labels: Dict[str, str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.context is None:
            self.context = {}


@dataclass
class Dashboard:
    """Monitoring dashboard configuration."""
    dashboard_id: str
    title: str
    description: str
    panels: List[Dict[str, Any]]
    refresh_interval_seconds: int = 30
    time_range_hours: int = 24
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, channel_id: str, config: Dict[str, Any]):
        self.channel_id = channel_id
        self.config = config
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification."""
        raise NotImplementedError


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel."""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to logs."""
        try:
            logger.warning(
                f"ALERT: {alert.title}",
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                metric=alert.metric_name,
                description=alert.description
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send log alert: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook-based notification channel."""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import aiohttp
            
            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                return False
            
            payload = {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "metric": alert.metric_name,
                "triggered_at": alert.triggered_at,
                "labels": alert.labels,
                "context": alert.context
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AdvancedMonitoringSystem:
    """Comprehensive monitoring and alerting system."""
    
    def __init__(self):
        # Metrics storage
        self.metrics: Dict[str, Metric] = {}
        self.metric_data_retention_hours = 168  # 7 days
        
        # Alerting
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.rule_states: Dict[str, Dict[str, Any]] = {}  # Track rule evaluation state
        
        # Notification channels
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Dashboards
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Background tasks
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.monitoring_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "alerts_resolved": 0,
            "notifications_sent": 0,
            "notification_failures": 0,
            "rule_evaluations": 0
        }
        
        # Initialize default notification channels
        self._initialize_default_channels()
        
        # Initialize system metrics
        self._initialize_system_metrics()
    
    def _initialize_default_channels(self):
        """Initialize default notification channels."""
        self.notification_channels["log"] = LogNotificationChannel("log", {})
    
    def _initialize_system_metrics(self):
        """Initialize built-in system metrics."""
        system_metrics = [
            ("system_cpu_usage", MetricType.GAUGE, "System CPU usage percentage", "%"),
            ("system_memory_usage", MetricType.GAUGE, "System memory usage percentage", "%"),
            ("system_disk_usage", MetricType.GAUGE, "System disk usage percentage", "%"),
            ("http_requests_total", MetricType.COUNTER, "Total HTTP requests", "requests"),
            ("http_request_duration", MetricType.HISTOGRAM, "HTTP request duration", "seconds"),
            ("database_connections_active", MetricType.GAUGE, "Active database connections", "connections"),
            ("database_query_duration", MetricType.HISTOGRAM, "Database query duration", "seconds"),
            ("cache_hit_rate", MetricType.GAUGE, "Cache hit rate", "percentage"),
            ("error_rate", MetricType.GAUGE, "Error rate", "percentage"),
            ("response_time_95th", MetricType.GAUGE, "95th percentile response time", "ms")
        ]
        
        for name, metric_type, description, unit in system_metrics:
            self.register_metric(name, metric_type, description, unit)
    
    def register_metric(self, name: str, metric_type: MetricType, 
                       description: str, unit: str, labels: Set[str] = None) -> Metric:
        """Register a new metric."""
        if labels is None:
            labels = set()
        
        metric = Metric(
            name=name,
            type=metric_type,
            description=description,
            unit=unit,
            labels=labels,
            data_points=deque(maxlen=10000)
        )
        
        self.metrics[name] = metric
        logger.debug(f"Registered metric: {name} ({metric_type.value})")
        return metric
    
    def record_metric(self, name: str, value: Union[float, int], 
                     labels: Dict[str, str] = None, timestamp: Optional[float] = None):
        """Record a metric value."""
        if name not in self.metrics:
            logger.warning(f"Metric not registered: {name}")
            return
        
        if labels is None:
            labels = {}
        
        if timestamp is None:
            timestamp = time.time()
        
        point = MetricPoint(timestamp=timestamp, value=value, labels=labels)
        self.metrics[name].data_points.append(point)
        
        self.monitoring_stats["metrics_collected"] += 1
    
    def get_metric_data(self, name: str, start_time: Optional[float] = None,
                       end_time: Optional[float] = None, 
                       labels_filter: Dict[str, str] = None) -> List[MetricPoint]:
        """Get metric data points within time range."""
        if name not in self.metrics:
            return []
        
        metric = self.metrics[name]
        now = time.time()
        
        if start_time is None:
            start_time = now - 3600  # Last hour by default
        if end_time is None:
            end_time = now
        
        # Filter by time range and labels
        filtered_points = []
        for point in metric.data_points:
            if start_time <= point.timestamp <= end_time:
                # Apply label filter if specified
                if labels_filter:
                    if all(point.labels.get(k) == v for k, v in labels_filter.items()):
                        filtered_points.append(point)
                else:
                    filtered_points.append(point)
        
        return filtered_points
    
    def register_alert_rule(self, rule: AlertRule):
        """Register an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        self.rule_states[rule.rule_id] = {
            "last_evaluation": 0,
            "condition_true_since": None,
            "last_alert_time": 0,
            "consecutive_violations": 0
        }
        
        logger.info(f"Registered alert rule: {rule.name} ({rule.rule_id})")
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        # Start alert evaluation task
        async def alert_evaluation_loop():
            while True:
                try:
                    await self._evaluate_alert_rules()
                    await asyncio.sleep(10)  # Evaluate every 10 seconds
                except Exception as e:
                    logger.error(f"Alert evaluation error: {e}")
                    await asyncio.sleep(30)
        
        # Start metric collection task
        async def metric_collection_loop():
            while True:
                try:
                    await self._collect_system_metrics()
                    await asyncio.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    logger.error(f"Metric collection error: {e}")
                    await asyncio.sleep(60)
        
        # Start cleanup task
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_old_data()
                    await asyncio.sleep(3600)  # Cleanup every hour
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                    await asyncio.sleep(1800)  # Wait 30 min on error
        
        self._monitoring_tasks["alert_evaluation"] = asyncio.create_task(alert_evaluation_loop())
        self._monitoring_tasks["metric_collection"] = asyncio.create_task(metric_collection_loop())
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        
        logger.info("Advanced monitoring system started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        for task_name, task in self._monitoring_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug(f"Stopped monitoring task: {task_name}")
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._monitoring_tasks.clear()
        logger.info("Advanced monitoring system stopped")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system_cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_usage", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("system_disk_usage", disk_percent)
            
        except ImportError:
            # psutil not available, collect basic metrics
            pass
        except Exception as e:
            logger.debug(f"System metrics collection error: {e}")
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        now = time.time()
        
        for rule_id, rule in self.alert_rules.items():
            try:
                await self._evaluate_rule(rule_id, rule, now)
                self.monitoring_stats["rule_evaluations"] += 1
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {e}")
    
    async def _evaluate_rule(self, rule_id: str, rule: AlertRule, now: float):
        """Evaluate a single alert rule."""
        state = self.rule_states[rule_id]
        
        # Get recent metric data
        metric_data = self.get_metric_data(
            rule.metric_name,
            start_time=now - rule.duration_seconds,
            end_time=now,
            labels_filter=rule.labels if rule.labels else None
        )
        
        if not metric_data:
            return
        
        # Check condition
        condition_met = self._check_condition(metric_data, rule.condition)
        
        if condition_met:
            if state["condition_true_since"] is None:
                state["condition_true_since"] = now
            
            # Check if condition has been true long enough
            duration_met = (now - state["condition_true_since"]) >= rule.duration_seconds
            
            # Check cooldown period
            cooldown_expired = (now - state["last_alert_time"]) >= rule.cooldown_seconds
            
            if duration_met and cooldown_expired:
                await self._trigger_alert(rule, metric_data, now)
                state["last_alert_time"] = now
                state["consecutive_violations"] += 1
        else:
            # Condition no longer met
            state["condition_true_since"] = None
            state["consecutive_violations"] = 0
            
            # Auto-resolve alert if configured
            if rule.auto_resolve:
                await self._auto_resolve_alerts(rule_id)
        
        state["last_evaluation"] = now
    
    def _check_condition(self, metric_data: List[MetricPoint], condition: str) -> bool:
        """Check if condition is met for metric data."""
        if not metric_data:
            return False
        
        # Get the latest value
        latest_value = metric_data[-1].value
        
        try:
            # Parse condition (e.g., "> 0.8", "< 100", "== 0")
            operator = condition.split()[0]
            threshold = float(condition.split()[1])
            
            if operator == ">":
                return latest_value > threshold
            elif operator == "<":
                return latest_value < threshold
            elif operator == ">=":
                return latest_value >= threshold
            elif operator == "<=":
                return latest_value <= threshold
            elif operator == "==":
                return latest_value == threshold
            elif operator == "!=":
                return latest_value != threshold
            else:
                logger.warning(f"Unknown condition operator: {operator}")
                return False
                
        except (IndexError, ValueError) as e:
            logger.error(f"Invalid condition format: {condition} - {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metric_data: List[MetricPoint], timestamp: float):
        """Trigger an alert."""
        alert_id = f"ALERT-{int(timestamp)}-{rule.rule_id}"
        
        # Check if alert already exists
        if alert_id in self.active_alerts:
            return
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=f"{rule.name}",
            description=rule.description,
            triggered_at=datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            labels=rule.labels.copy(),
            context={
                "condition": rule.condition,
                "current_value": metric_data[-1].value if metric_data else None,
                "metric_points_count": len(metric_data)
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.monitoring_stats["alerts_triggered"] += 1
        
        # Send notifications
        await self._send_alert_notifications(alert, rule.notification_channels)
        
        logger.warning(
            f"Alert triggered: {alert.title}",
            alert_id=alert_id,
            severity=rule.severity.value,
            metric=rule.metric_name,
            value=alert.context.get("current_value")
        )
    
    async def _send_alert_notifications(self, alert: Alert, channels: List[str]):
        """Send alert notifications to specified channels."""
        if not channels:
            channels = ["log"]  # Default to log channel
        
        for channel_id in channels:
            if channel_id in self.notification_channels:
                try:
                    success = await self.notification_channels[channel_id].send_alert(alert)
                    if success:
                        self.monitoring_stats["notifications_sent"] += 1
                    else:
                        self.monitoring_stats["notification_failures"] += 1
                except Exception as e:
                    logger.error(f"Notification channel {channel_id} failed: {e}")
                    self.monitoring_stats["notification_failures"] += 1
    
    async def _auto_resolve_alerts(self, rule_id: str):
        """Auto-resolve alerts for a rule when condition is no longer met."""
        resolved_count = 0
        now = datetime.now(timezone.utc).isoformat()
        
        for alert_id, alert in list(self.active_alerts.items()):
            if (alert.rule_id == rule_id and 
                alert.status == AlertStatus.ACTIVE):
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = now
                del self.active_alerts[alert_id]
                resolved_count += 1
                
                logger.info(f"Auto-resolved alert: {alert.title} ({alert_id})")
        
        if resolved_count > 0:
            self.monitoring_stats["alerts_resolved"] += resolved_count
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc).isoformat()
        alert.context["acknowledged_by"] = acknowledged_by
        
        logger.info(f"Alert acknowledged: {alert.title} by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Manually resolve an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc).isoformat()
        alert.context["resolved_by"] = resolved_by
        
        del self.active_alerts[alert_id]
        self.monitoring_stats["alerts_resolved"] += 1
        
        logger.info(f"Alert resolved: {alert.title} by {resolved_by}")
        return True
    
    def add_notification_channel(self, channel_id: str, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels[channel_id] = channel
        logger.info(f"Added notification channel: {channel_id}")
    
    def create_dashboard(self, dashboard: Dashboard):
        """Create a monitoring dashboard."""
        self.dashboards[dashboard.dashboard_id] = dashboard
        logger.info(f"Created dashboard: {dashboard.title}")
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data."""
        if dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard = self.dashboards[dashboard_id]
        now = time.time()
        start_time = now - (dashboard.time_range_hours * 3600)
        
        dashboard_data = {
            "dashboard_id": dashboard_id,
            "title": dashboard.title,
            "description": dashboard.description,
            "time_range": {"start": start_time, "end": now},
            "panels": []
        }
        
        # Generate data for each panel
        for panel in dashboard.panels:
            panel_data = {
                "title": panel.get("title", ""),
                "type": panel.get("type", "graph"),
                "metrics": []
            }
            
            for metric_name in panel.get("metrics", []):
                metric_points = self.get_metric_data(metric_name, start_time, now)
                panel_data["metrics"].append({
                    "name": metric_name,
                    "data": [point.to_dict() for point in metric_points]
                })
            
            dashboard_data["panels"].append(panel_data)
        
        return dashboard_data
    
    def get_monitoring_overview(self) -> Dict[str, Any]:
        """Get system monitoring overview."""
        active_alerts_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_alerts_by_severity[alert.severity.value] += 1
        
        return {
            "metrics": {
                "total_registered": len(self.metrics),
                "data_points_stored": sum(len(m.data_points) for m in self.metrics.values()),
                "metrics_collected_24h": self.monitoring_stats["metrics_collected"]
            },
            "alerts": {
                "active_total": len(self.active_alerts),
                "active_by_severity": dict(active_alerts_by_severity),
                "triggered_24h": self.monitoring_stats["alerts_triggered"],
                "resolved_24h": self.monitoring_stats["alerts_resolved"]
            },
            "alerting": {
                "rules_configured": len(self.alert_rules),
                "notifications_sent": self.monitoring_stats["notifications_sent"],
                "notification_failures": self.monitoring_stats["notification_failures"]
            },
            "system": {
                "monitoring_tasks_active": len(self._monitoring_tasks),
                "dashboards_configured": len(self.dashboards),
                "notification_channels": len(self.notification_channels)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _cleanup_old_data(self):
        """Clean up old metric data and alert history."""
        cutoff_time = time.time() - (self.metric_data_retention_hours * 3600)
        cleaned_points = 0
        
        # Clean metric data
        for metric in self.metrics.values():
            original_count = len(metric.data_points)
            # Remove old points (keeping data structure efficient)
            while metric.data_points and metric.data_points[0].timestamp < cutoff_time:
                metric.data_points.popleft()
                cleaned_points += 1
        
        # Clean old alert history (keep for 30 days)
        alert_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        original_alert_count = len(self.alert_history)
        
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.triggered_at) > alert_cutoff
        ]
        
        cleaned_alerts = original_alert_count - len(self.alert_history)
        
        if cleaned_points > 0 or cleaned_alerts > 0:
            logger.debug(f"Cleaned up {cleaned_points} old metric points and {cleaned_alerts} old alerts")


# Global monitoring system instance
_monitoring_system: Optional[AdvancedMonitoringSystem] = None


def get_monitoring_system() -> AdvancedMonitoringSystem:
    """Get or create the global monitoring system."""
    global _monitoring_system
    
    if _monitoring_system is None:
        _monitoring_system = AdvancedMonitoringSystem()
    
    return _monitoring_system


# Convenience functions for common monitoring tasks
def record_http_request(duration_ms: float, status_code: int, endpoint: str):
    """Record HTTP request metrics."""
    monitoring = get_monitoring_system()
    labels = {"status_code": str(status_code), "endpoint": endpoint}
    
    monitoring.record_metric("http_requests_total", 1, labels)
    monitoring.record_metric("http_request_duration", duration_ms / 1000, labels)


def record_database_query(duration_ms: float, database: str, operation: str, success: bool):
    """Record database query metrics."""
    monitoring = get_monitoring_system()
    labels = {
        "database": database,
        "operation": operation,
        "success": str(success).lower()
    }
    
    monitoring.record_metric("database_query_duration", duration_ms / 1000, labels)


def record_cache_access(hit: bool, cache_type: str = "default"):
    """Record cache access metrics."""
    monitoring = get_monitoring_system()
    labels = {"cache_type": cache_type, "result": "hit" if hit else "miss"}
    
    monitoring.record_metric("cache_hit_rate", 1 if hit else 0, labels)