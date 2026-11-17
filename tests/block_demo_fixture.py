"""分块指标组测试夹具：本模块按约定不导入pandas。"""

from __future__ import annotations

import numpy as np

from trans_to_min import kernels
from trans_to_min.blocks import BlockMetricGroup, CellValues


class DemoBlockGroup(BlockMetricGroup):
    """笔数、量、去重订单数与已知金额四个示例指标。"""

    def compute_block(self, block, metric_names):
        table = block.tables["chengjiao"]
        valid = table["minute"] >= 0
        cells = block.cells("chengjiao", valid)
        volume = table["volume"][valid]
        price = table["price"][valid]
        orders = table["order_num"][valid]
        known = ~np.isnan(price)
        result = {
            "block_count_min": CellValues(*kernels.count_by(cells)),
            "block_volume_min": CellValues(*kernels.sum_by(cells, volume)),
            "block_orders_min": CellValues(*kernels.distinct_count_by(cells, orders)),
            "block_money_min": CellValues(*kernels.sum_by(
                cells[known], price[known] * volume[known].astype(np.float64))),
        }
        return {name: result[name] for name in metric_names}


class DuplicateKeys(DemoBlockGroup):
    def compute_block(self, block, metric_names):
        return {name: CellValues(np.array([0, 0]), np.array([1, 2], dtype=np.uint64))
                for name in metric_names}


class MissingMetric(DemoBlockGroup):
    def compute_block(self, block, metric_names):
        return {}


class ListResult(DemoBlockGroup):
    def compute_block(self, block, metric_names):
        return {name: CellValues([0], [1]) for name in metric_names}
