"""经真实Parquet读取，防止数量与订单号绕回、截断、溢出或丢失事件。"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trans_to_min import L2InputAdapter, input_path


FIELDS = (
    ("weituo", "volume"),
    ("weituo", "order_num"),
    ("chengjiao", "buy_order_num"),
    ("chengjiao", "sell_order_num"),
)


@pytest.mark.parametrize("dataset,column", FIELDS)
@pytest.mark.parametrize(
    "physical_type,values,expected",
    [
        pytest.param(pa.uint32(), [0, 7, 4294967295], [0, 7, 4294967295], id="uint32"),
        pytest.param(
            pa.uint64(),
            [0, 9223372036854775808, 18446744073709551615],
            [0, 9223372036854775808, 18446744073709551615],
            id="uint64",
        ),
        pytest.param(pa.int64(), [0, 9223372036854775807], [0, 9223372036854775807], id="int64"),
        pytest.param(pa.float64(), [0.0, 7.0], [0, 7], id="float64"),
        pytest.param(
            pa.float64(), [18446744073709549568.0], [18446744073709549568],
            id="float64_below_upper_bound",
        ),
        pytest.param(pa.uint32(), [], [], id="empty_uint32"),
        pytest.param(pa.float64(), [], [], id="empty_float64"),
        pytest.param(pa.bool_(), [True, False], [1, 0], id="bool"),
    ],
)
def test_read_unsigned_preserves_exact_values_and_event_order(
    tmp_path: Path, dataset: str, column: str, physical_type, values, expected,
) -> None:
    """合法物理类型均转uint64，尤其不得损失高位整数或改变行序。"""

    path = input_path(tmp_path, dataset, 20260928)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({column: pa.array(values, type=physical_type)}), path)

    result = L2InputAdapter(tmp_path).read(dataset, 20260928, (column,))

    assert str(result[column].dtype) == "uint64"
    assert len(result) == len(values)
    assert result.columns.tolist() == [column]
    assert result[column].tolist() == expected


@pytest.mark.parametrize("dataset,column", FIELDS)
@pytest.mark.parametrize(
    "physical_type,values",
    [
        pytest.param(pa.int64(), [5, -5], id="negative_signed"),
        pytest.param(pa.float64(), [1, 1.7], id="fractional_float"),
        pytest.param(pa.float64(), [1, -1], id="negative_float"),
        pytest.param(pa.float64(), [1, float("nan")], id="nan_float"),
        pytest.param(pa.float64(), [1, float("inf")], id="positive_inf_float"),
        pytest.param(pa.float64(), [1, float("-inf")], id="negative_inf_float"),
        pytest.param(pa.float64(), [1, 18446744073709551616.0], id="overflow_float"),
        pytest.param(pa.int64(), [1, None], id="null_int64"),
        pytest.param(pa.uint32(), [1, None], id="null_uint32"),
        pytest.param(pa.uint64(), [1, None], id="null_uint64"),
        pytest.param(pa.uint64(), [18446744073709551615, None], id="null_uint64_max"),
    ],
)
def test_read_unsigned_rejects_invalid_source_with_field_context(
    tmp_path: Path, dataset: str, column: str, physical_type, values,
) -> None:
    """公开读取必须拒绝非法事件，错误须定位真实数据集和字段。"""

    path = input_path(tmp_path, dataset, 20260928)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({column: pa.array(values, type=physical_type)}), path)

    with pytest.raises(ValueError) as error:
        L2InputAdapter(tmp_path).read(dataset, 20260928, (column,))

    assert f"{dataset}.{column}" in str(error.value)
