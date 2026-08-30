"""
Dashboard Module
Terminal-based visualization of training metrics
"""

from typing import List, Optional
from sentinel.cli.job_tracker import TrainingJob, TrainingMetrics, MetricsCollector


class TerminalDashboard:
    """Terminal-based training dashboard"""

    def __init__(self, job: TrainingJob, metrics_collector: MetricsCollector):
        """Initialize dashboard"""
        self.job = job
        self.metrics = metrics_collector

    def render_header(self) -> str:
        """Render dashboard header"""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("🎯 Training Dashboard".center(70))
        lines.append("="*70)
        return "\n".join(lines)

    def render_job_info(self) -> str:
        """Render job information section"""
        lines = []
        lines.append("\n📋 Job Information:")
        lines.append(f"   Job ID:        {self.job.job_id}")
        lines.append(f"   Model:         {self.job.model_name}")
        lines.append(f"   Dataset:       {self.job.dataset_path}")
        lines.append(f"   Status:        {self.job.status.value.upper()}")
        lines.append(f"   Epochs:        {self.job.current_epoch}/{self.job.total_epochs}")
        lines.append(f"   Batch Size:    {self.job.batch_size}")
        lines.append(f"   Learning Rate: {self.job.learning_rate}")
        lines.append(f"   GPU Enabled:   {'Yes' if self.job.gpu_enabled else 'No'}")
        return "\n".join(lines)

    def render_progress_bar(self, width: int = 50) -> str:
        """Render training progress bar"""
        progress = int((self.job.progress_percent / 100.0) * width)
        bar = "█" * progress + "░" * (width - progress)
        return f"   [{bar}] {self.job.progress_percent:.1f}%"

    def render_metrics(self) -> str:
        """Render current metrics"""
        lines = []
        latest = self.metrics.get_latest_metric()

        if not latest:
            return "   No metrics yet..."

        lines.append("\n📊 Current Metrics:")
        lines.append(f"   Loss:        {latest.loss:.4f}")
        lines.append(f"   Accuracy:    {latest.accuracy:.4f} ({latest.accuracy*100:.2f}%)")
        lines.append(f"   Val Loss:    {latest.val_loss:.4f}")
        lines.append(f"   Val Accuracy: {latest.val_accuracy:.4f} ({latest.val_accuracy*100:.2f}%)")

        # Show best seen
        lines.append("\n🏆 Best Seen:")
        lines.append(f"   Best Loss:     {self.metrics.get_best_loss():.4f}")
        lines.append(f"   Best Accuracy: {self.metrics.get_best_accuracy():.4f} ({self.metrics.get_best_accuracy()*100:.2f}%)")

        # Show trends
        lines.append("\n📈 Trends:")
        lines.append(f"   Loss:     {self.metrics.get_trend('loss')}")
        lines.append(f"   Accuracy: {self.metrics.get_trend('accuracy')}")

        return "\n".join(lines)

    def render_chart(self, metric_name: str = "loss", width: int = 60,
                    height: int = 10) -> str:
        """Render simple ASCII chart of metric"""
        metrics = self.metrics.get_metrics()
        if not metrics:
            return "   No data to plot"

        # Get metric values
        if metric_name == "loss":
            values = [m.loss for m in metrics]
        elif metric_name == "accuracy":
            values = [m.accuracy for m in metrics]
        elif metric_name == "val_loss":
            values = [m.val_loss for m in metrics]
        elif metric_name == "val_accuracy":
            values = [m.val_accuracy for m in metrics]
        else:
            return "   Unknown metric"

        if not values:
            return "   No data to plot"

        # Normalize values to chart height
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1

        # Only show last width values
        values_to_plot = values[-width:]

        # Build chart from bottom to top
        lines = []
        lines.append(f"\n📈 {metric_name.upper()} Chart:")

        for row in range(height, 0, -1):
            line = "   "
            threshold = min_val + (range_val * row / height)

            for val in values_to_plot:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "

            lines.append(line)

        # Add baseline
        lines.append("   " + "─" * len(values_to_plot))

        return "\n".join(lines)

    def render_footer(self) -> str:
        """Render dashboard footer"""
        lines = []
        lines.append("="*70 + "\n")
        return "\n".join(lines)

    def render_full(self, show_chart: bool = True) -> str:
        """Render complete dashboard"""
        parts = []

        parts.append(self.render_header())
        parts.append(self.render_job_info())
        parts.append(f"\n⏳ Progress:")
        parts.append(self.render_progress_bar())
        parts.append(self.render_metrics())

        if show_chart:
            parts.append(self.render_chart("loss", width=50, height=8))
            parts.append(self.render_chart("accuracy", width=50, height=8))

        parts.append(self.render_footer())

        return "\n".join(parts)


class MetricsFormatter:
    """Format metrics for display"""

    @staticmethod
    def format_metric(value: float, decimals: int = 4) -> str:
        """Format metric value"""
        return f"{value:.{decimals}f}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format as percentage"""
        return f"{value*100:.{decimals}f}%"

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time duration"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @staticmethod
    def format_summary(metrics: List[TrainingMetrics]) -> str:
        """Format metrics summary"""
        if not metrics:
            return "No metrics available"

        first = metrics[0]
        latest = metrics[-1]

        lines = []
        lines.append("Training Summary:")
        lines.append(f"  Epochs: {len(metrics)}")
        lines.append(f"  Initial Loss: {first.loss:.4f} → Final Loss: {latest.loss:.4f}")
        lines.append(f"  Initial Acc:  {first.accuracy:.4f} → Final Acc: {latest.accuracy:.4f}")

        # Calculate improvement
        loss_improvement = first.loss - latest.loss
        acc_improvement = latest.accuracy - first.accuracy

        if loss_improvement > 0:
            lines.append(f"  Loss improved by: {loss_improvement:.4f} ↓")
        if acc_improvement > 0:
            lines.append(f"  Accuracy improved by: {acc_improvement:.4f} ↑")

        return "\n".join(lines)


class ProgressIndicator:
    """Simple progress indicator for terminal"""

    def __init__(self, total: int, desc: str = "Progress"):
        """Initialize progress indicator"""
        self.total = total
        self.desc = desc
        self.current = 0

    def update(self, current: int):
        """Update progress"""
        self.current = current

    def render(self, width: int = 30) -> str:
        """Render progress indicator"""
        if self.total == 0:
            return "0% (0/0)"

        progress = int((self.current / self.total) * width)
        bar = "█" * progress + "░" * (width - progress)
        percent = (self.current / self.total) * 100
        return f"[{bar}] {percent:.1f}% ({self.current}/{self.total})"
