"""
Enhanced logging utility for KSB Assessment with performance tracking and metrics.
"""
import time
import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log verbosity levels."""
    MINIMAL = "minimal"      # Only critical events and errors
    STANDARD = "standard"    # Key milestones and warnings
    VERBOSE = "verbose"      # Everything (debugging)


class PerformanceTimer:
    """Track operation timing and emit warnings for slow operations."""

    def __init__(self, operation: str, warn_threshold_ms: float = 3000):
        self.operation = operation
        self.warn_threshold_ms = warn_threshold_ms
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000

        if duration_ms > self.warn_threshold_ms:
            logger.warning(
                f"⚠️ Slow operation: {self.operation} took {duration_ms:.0f}ms "
                f"(threshold: {self.warn_threshold_ms:.0f}ms)"
            )
        else:
            logger.debug(f"✓ {self.operation} completed in {duration_ms:.0f}ms")

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        elif self.start_time:
            return (time.time() - self.start_time) * 1000
        return 0


class AssessmentLogger:
    """Enhanced logger for KSB assessment with context and metrics."""

    def __init__(self, name: str, level: LogLevel = LogLevel.STANDARD, verbose: bool = False):
        self.name = name
        # If verbose is True, override level to VERBOSE
        self.level = LogLevel.VERBOSE if verbose else level
        self.metrics: Dict[str, Any] = {}
        self.timers: Dict[str, PerformanceTimer] = {}

    def set_level(self, level: LogLevel):
        """Update log level."""
        self.level = level

    def _should_log(self, required_level: LogLevel) -> bool:
        """Check if message should be logged based on current level."""
        hierarchy = {
            LogLevel.MINIMAL: 0,
            LogLevel.STANDARD: 1,
            LogLevel.VERBOSE: 2
        }
        return hierarchy[self.level] >= hierarchy[required_level]

    def info(self, message: str, level: LogLevel = LogLevel.STANDARD):
        """Log info message if level permits."""
        if self._should_log(level):
            logger.info(f"[{self.name}] {message}")

    def debug(self, message: str):
        """Log debug message (verbose only)."""
        if self._should_log(LogLevel.VERBOSE):
            logger.debug(f"[{self.name}] {message}")

    def warning(self, message: str, level: LogLevel = LogLevel.MINIMAL):
        """Log warning (always shown unless minimal)."""
        if self._should_log(level):
            logger.warning(f"[{self.name}] ⚠️ {message}")

    def error(self, message: str, exc: Optional[Exception] = None):
        """Log error (always shown)."""
        if exc:
            logger.error(f"[{self.name}] ❌ {message}: {exc}")
        else:
            logger.error(f"[{self.name}] ❌ {message}")

    def phase(self, message: str):
        """Log phase transition (standard+)."""
        if self._should_log(LogLevel.STANDARD):
            logger.info(f"[{self.name}] 🔄 {message}")

    def success(self, message: str):
        """Log success (standard+)."""
        if self._should_log(LogLevel.STANDARD):
            logger.info(f"[{self.name}] ✅ {message}")

    def metric(self, key: str, value: Any):
        """Record a metric."""
        self.metrics[key] = value
        self.debug(f"📊 {key}: {value}")

    @contextmanager
    def timer(self, operation: str, warn_threshold_ms: float = 3000):
        """Time an operation and warn if slow."""
        timer = PerformanceTimer(operation, warn_threshold_ms)
        self.timers[operation] = timer

        with timer:
            yield timer

        # Log timing at appropriate level
        duration_ms = timer.elapsed_ms()
        if duration_ms > warn_threshold_ms:
            self.warning(f"{operation} took {duration_ms:.0f}ms", LogLevel.STANDARD)
        else:
            self.debug(f"{operation} completed in {duration_ms:.0f}ms")

    def progress(self, current: int, total: int, item_name: str = "item"):
        """Log progress (verbose only)."""
        if self._should_log(LogLevel.VERBOSE):
            pct = (current / total * 100) if total > 0 else 0
            logger.info(f"[{self.name}] 📈 Processing {item_name} {current}/{total} ({pct:.0f}%)")

    def evidence_stats(self, ksb_code: str, chunks_found: int,
                      strategy: str, query_variations: int,
                      ocr_chunks: int = 0, avg_similarity: Optional[float] = None):
        """Log evidence search statistics."""
        msg = f"{ksb_code}: {chunks_found} chunks ({strategy}, {query_variations} queries"

        if ocr_chunks > 0:
            pct = (ocr_chunks / chunks_found * 100) if chunks_found > 0 else 0
            msg += f", {ocr_chunks} from OCR [{pct:.0f}%]"

        if avg_similarity is not None:
            msg += f", avg sim: {avg_similarity:.2f}"

        msg += ")"

        # Warn if too few chunks found
        if chunks_found < 3:
            self.warning(f"{ksb_code}: Only {chunks_found} chunks found - may lack evidence", LogLevel.STANDARD)

        self.debug(f"  {msg}")

    def ocr_result(self, image_id: str, chars_extracted: int, duration_ms: float):
        """Log OCR extraction result."""
        if chars_extracted > 0:
            self.debug(f"  OCR extracted {chars_extracted} chars from {image_id} ({duration_ms:.0f}ms)")
        else:
            self.debug(f"  OCR: No text extracted from {image_id}")

        if duration_ms > 3000:
            self.warning(f"Slow OCR: {image_id} took {duration_ms:.0f}ms", LogLevel.STANDARD)

    def grade_decision(self, ksb_code: str, grade: str, confidence: str,
                      evidence_count: int, criteria_met: bool):
        """Log grading decision with context."""
        msg = f"{ksb_code}: {grade} (confidence: {confidence}, evidence: {evidence_count} chunks, criteria met: {criteria_met})"

        # Warn on low-confidence REFERRAL
        if grade == "REFERRAL" and confidence == "LOW":
            self.warning(f"{ksb_code}: Low-confidence REFERRAL - may need review", LogLevel.STANDARD)

        self.debug(f"  {msg}")

    def get_summary(self) -> str:
        """Get summary of logged metrics."""
        lines = [f"\n{'='*60}", f"Assessment Metrics: {self.name}", f"{'='*60}"]

        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value}")

        # Add timing summary
        if self.timers:
            lines.append("\nOperation Timings:")
            for op, timer in self.timers.items():
                lines.append(f"  {op}: {timer.elapsed_ms():.0f}ms")

        lines.append("=" * 60)
        return "\n".join(lines)


def create_logger(name: str, level: LogLevel = LogLevel.STANDARD,
                 verbose: bool = False) -> AssessmentLogger:
    """Factory function to create an AssessmentLogger."""
    return AssessmentLogger(name, level, verbose)
