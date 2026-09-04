"""逐笔数据转分钟数据框架。"""

from .adapter import L2InputAdapter, input_path
from ._version import __version__
from .config import RunConfig
from .group import MinuteMetricGroup
from .group_runner import MinuteMetricGroupRunner
from .metric import MinuteMetric
from .resources import ParallelismRecommendation, recommend_date_workers
from .runner import MinuteConversionRunner

__all__ = [
    "__version__",
    "L2InputAdapter",
    "MinuteConversionRunner",
    "MinuteMetricGroup",
    "MinuteMetricGroupRunner",
    "MinuteMetric",
    "ParallelismRecommendation",
    "RunConfig",
    "input_path",
    "recommend_date_workers",
]
