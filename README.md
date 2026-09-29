# Trans_To_Min：逐笔数据转分钟数据框架

`Trans_To_Min`把按交易日保存的A股逐笔委托、成交和撤单Parquet转换为自定义分钟数据。
框架负责输入适配、分钟切片、股票与分钟补齐、并发调度、增量运行和按月发布；具体
数据的研究含义和计算公式由独立业务项目实现。

当前稳定版本为`1.0.1`。这是后续业务项目依赖、生产结果追溯和兼容性判断的当前
生产版本。版本变更见[CHANGELOG.md](CHANGELOG.md)。

## 1. 核心能力

- 读取`weituo`、`chengjiao`、`chedan`三类按日Parquet，只加载指标声明的列；
- 兼容压缩字段、价格与金额缩放、整数类型编码以及`date + time`时间字段；
- 固定生成本框架定义的242个分钟结束标签；
- 使用左开右闭区间，并明确规定09:30、11:30、13:00和15:00的特殊边界；
- 对指标实际输入中出现的股票取并集，并为这些股票补齐242个分钟；
- 支持`float64`、`int64`和`uint64`，包括整数指标的Parquet空值；
- 按交易日多进程计算，并约束每个进程的PyArrow线程数；
- 根据CPU、可用内存和Parquet列体积自动选择并发，并按月用真实峰值校准；
- 把日分片流式合并为月文件，支持新增日期、显式覆盖和断点续跑；
- 月文件使用临时文件校验后原子替换，避免暴露半写入文件；
- 支持同源多指标共享一次物理读取和可选的业务中间计算；
- 每个最终指标仍独立保存Schema、README、月份文件和`manifest.json`。

## 2. 功能边界

框架只提供通用转换基础设施，不包含以下内容：

- 不定义正式业务指标、因子、信息熵或研究结论；
- 不自动读取股票池、上市状态、ST状态、停牌表或退市表；
- 不为完全没有相关逐笔记录的股票凭空创建分钟行；
- 不连接`cn_stock_1min`，也不把数据库校验写入生产转换流程；
- 不负责数据库入库、任务调度、监控告警和集群级分布式计算；
- 不自动决定订单关联键、主动方向、价格取最大或最小等业务公式；
- 不实现新交易制度下独立的盘后半小时分钟轴；当前盘前和盘后处理遵循第6节的
  固定边界；
- 不保证源逐笔数据的业务正确性，输入质量和结果经济含义必须由业务项目验证。

## 3. 安装与使用方式

### 3.1 环境要求

- Python `3.11`及以上；
- NumPy `1.26`及以上且低于`3`；
- Pandas `2.1`及以上且低于`3`；
- PyArrow `14`及以上且低于`25`。

### 3.2 服务器推荐安装

服务器框架目录为`/app/workspace/zhangyuan/Trans_To_Min`，推荐安装到公共虚拟环境：

```bash
/app/workspace/zhangyuan/.venv/bin/pip install -e \
  /app/workspace/zhangyuan/Trans_To_Min
```

`-e`表示可编辑安装。框架目录更新后，业务项目通常无需再次复制或安装源码。

普通非可编辑安装也可使用：

```bash
python -m pip install /app/workspace/zhangyuan/Trans_To_Min
```

验证安装和版本：

```bash
python -c "import trans_to_min; print(trans_to_min.__version__)"
```

### 3.3 不安装直接使用

直接复制也能运行，但必须满足以下一种条件：

- 业务脚本从本仓库根目录启动；
- 把完整`trans_to_min/`包复制到业务脚本可导入的位置；
- 把本仓库根目录加入`PYTHONPATH`。

生产环境推荐可编辑安装。复制包容易产生多个版本，导致不同业务任务使用的框架
口径不一致。

## 4. 项目与部署目录

框架和业务数据任务分开维护：

| 内容 | 本地目录 | 服务器目录 |
|---|---|---|
| 通用框架 | `D:\hytp\Trans_To_Min` | `/app/workspace/zhangyuan/Trans_To_Min` |
| 数据任务根目录 | `D:\hytp\逐笔转分钟数据` | `/app/workspace/zhangyuan/逐笔转分钟数据` |
| 逐笔输入 | — | `/sd1-data/level2` |
| 分钟输出 | — | `/data/zhangyuan/trans_to_min` |

本仓库结构：

```text
Trans_To_Min/
├── trans_to_min/                   # 可安装Python包
│   ├── adapter.py                  # 输入路径、字段读取和归一化
│   ├── minute.py                   # 242分钟轴和分钟编码
│   ├── metric.py                   # 单指标抽象基类
│   ├── runner.py                   # 单指标执行器
│   ├── group.py                    # 同源指标组定义
│   ├── group_runner.py             # 同源指标组执行器
│   ├── resources.py                # CPU与内存感知并发建议
│   ├── storage.py                  # 日分片、月文件和manifest
│   └── metrics/trade_volume.py     # 框架参考指标
├── scripts/run_trade_volume.py     # 参考运行入口
├── tests/                          # 框架单元测试
├── pyproject.toml                  # 安装和依赖声明
└── CHANGELOG.md                    # 版本记录
```

正式指标不得直接堆入本仓库。每项业务数据应在“逐笔转分钟数据”任务根目录下建立
独立、自包含的任务目录，保存指标逻辑、运行入口、测试、校验和说明文档。任务目录名
必须使用小写英文字母、数字和下划线，并与该数据在
`/data/zhangyuan/trans_to_min/`下的主题目录名完全一致；中文仅用于任务总目录和
文档内容。例如生产主题目录为`time_center_min/`时，任务目录也应为
`逐笔转分钟数据/time_center_min/`。主题目录名称变更时，两处应同步调整。

## 5. 输入数据契约

### 5.1 文件路径

框架识别以下固定路径：

```text
/sd1-data/level2/{dataset}/{YYYY}/{YYYYMM}/{YYYYMMDD}_{dataset}.parquet
```

`dataset`只能是：

- `weituo`：委托；
- `chengjiao`：成交；
- `chedan`：撤单。

业务指标通过`inputs`声明需要哪些表，通过`input_columns`逐表声明需要哪些逻辑列。
框架不会读取未声明的业务列。

### 5.2 逻辑字段适配

`L2InputAdapter.read()`会按需执行以下适配：

- `time_ms`：把物理`time`解释为午夜以来毫秒数，输出`uint32`；
- `timestamp`：由物理`date + time`构造`Asia/Shanghai`时区时间戳；
- `trade_date`：由物理`date`构造交易日；
- `price`和`money`：整数编码时根据Parquet metadata中的`price_scale`和
  `money_scale`还原为实际数值；
- 当委托或撤单没有物理`money`列时，按逐条`price × volume`生成；
- `volume`、`order_num`、`buy_order_num`和`sell_order_num`归一化为`uint64`；
- 整数`type`根据metadata中的`type_encoding`转换为业务字符串；
- `symbol`保持`.XSHG/.XSHE`，并在内存中使用分类类型降低占用。

缺少指标声明的物理列、缩放metadata或类型编码时会立即报错，不做静默猜测。

### 5.3 日期发现

`discover_dataset_dates()`发现单个数据集的标准文件日期。
`validate_matching_date_sets()`只在指定的起止日期闭区间内检查多表日期完整性：

- 候选日期取所需输入表日期的并集；
- 区间内每个候选日期必须具有所有所需输入文件；
- 区间外的历史缺失不会阻止本次运行；
- 区间内无输入文件或任一所需表缺日时会在计算前报错。

## 6. 固定分钟口径

### 6.1 分钟标签

每天固定生成242个带`Asia/Shanghai`时区的分钟结束标签：

- 上午：09:30至11:30，共121个标签；
- 下午：13:00至15:00，共121个标签。

常规分钟使用左开右闭区间。例如`09:30:00.001`至`09:31:00.000`进入`09:31`
标签。框架输出的是区间结束时刻，不是区间开始时刻。

### 6.2 特殊边界

框架固定以下特殊边界，它们不是简单的交易时段截断：

- `include_pre_market=True`时，09:30及以前的事件统一进入09:30；关闭后仅
  09:30整点进入09:30；
- 11:29之后至11:31（含）的事件进入11:30；
- 12:59至13:00（含）的事件进入13:00；
- `include_post_market=True`时，14:59之后的事件统一进入15:00；关闭后只接收
  14:59之后至15:00（含）的事件；
- 其他午休记录不属于分钟轴并被过滤。

`MinuteMetric.include_pre_market`和`include_post_market`默认均为`True`。如果研究仅需
连续竞价中的09:30至14:57，可在结果读取或业务逻辑中进一步限制范围，但标准生产
文件仍保持242个标签，便于不同数据统一对齐。

### 6.3 股票集合与空分钟

- 股票集合取该指标声明输入表中当日实际出现股票的并集；
- 不区分正常交易、停牌或其他证券状态；
- 当日完全没有相关输入记录的股票不生成结果；
- 一旦股票在相关输入中出现，就补齐该股票当日全部242个分钟；
- `empty_value=0`适合成交量、笔数等可加总数据；
- `empty_value=None`适合价格、比例等需要保留缺失的数据；
- `int64/uint64 + empty_value=None`使用Pandas可空整数并写为Parquet空值。

## 7. 输出数据契约

### 7.1 命名和目录

指标名必须采用`{business_name}_min`格式：只能包含小写英文字母、数字和单下划线
分段，必须以字母开头并以`_min`结尾。路径分隔符、点、空格和路径逃逸均被拒绝。

生产输出采用“主题→类别→指标”三级结构：

```text
/data/zhangyuan/trans_to_min/
└── {topic_name}_min/
    └── {category_name}_min/                 # RunConfig.output_root
        └── {business_name}_min/
            ├── README.md                    # 指标业务说明
            ├── manifest.json                # 最近一次运行清单
            ├── {YYYY}/
            │   └── {YYYYMM}.parquet         # 正式月份文件
            └── .staging/                    # 单指标运行暂存目录
```

指标组的共享暂存位于`RunConfig.output_root/.group_staging/`，正式结果仍进入每项指标
自己的目录。

### 7.2 标准Schema

每个指标月文件只有一个业务值列：

| 字段 | Arrow类型 | 含义 |
|---|---|---|
| `trade_date` | `date32`，非空 | 交易日 |
| `minute` | `timestamp[ms, Asia/Shanghai]`，非空 | 左开右闭区间结束标签 |
| `symbol` | `string`，非空 | `.XSHG/.XSHE`股票代码 |
| `{metric_name}` | `float64/int64/uint64`，可空 | 指标值 |

唯一键是`trade_date + minute + symbol`。组内多个指标也禁止合并为生产宽表。

### 7.3 指标README

框架无法推断业务含义。每项正式指标必须在业务项目中维护：

```text
metric_docs/{business_name}_min/README.md
```

业务入口应在计算前把它原子同步到最终指标目录。指标README至少说明：名称与业务
含义、输入表及字段、数学公式、方向口径、关联键、分钟边界、股票范围、空值与异常
处理、输出类型和路径、生成与校验脚本、框架版本及验收结果。

### 7.4 manifest

`manifest.json`记录最近一次运行的：

- 状态、框架版本、指标名、运行编号和起止时间；
- 请求日期、待计算日期和跳过日期；
- 完整`RunConfig`；
- 请求并发、实际并发、估计依据和逐月校准历史；
- 每日行数、文件大小、股票数、耗时和进程峰值内存；
- 每月最终路径、行数、大小、日期数量和首末日期；
- 失败时的异常摘要。

manifest是运行审计信息，不替代业务指标README，也不是累计运行历史数据库。

## 8. 定义单个分钟指标

指标类继承`MinuteMetric`并声明输入、字段、输出类型和空分钟语义：

```python
from datetime import date

import pandas as pd

from trans_to_min import MinuteMetric


class MinuteTradeCount(MinuteMetric):
    """每只股票每分钟的逐笔成交笔数。"""

    name = "trade_count_min"
    inputs = ("chengjiao",)
    input_columns = {
        "chengjiao": ("symbol", "time_ms"),
    }
    value_type = "uint64"
    empty_value = 0

    def prepare_events(
        self,
        trade_date: date,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return inputs["chengjiao"].loc[:, ["symbol", "time_ms"]].copy()

    def aggregate_minutes(self, events: pd.DataFrame) -> pd.DataFrame:
        return (
            events.groupby(["symbol", "minute"], sort=False, observed=True)
            .size()
            .astype("uint64")
            .rename(self.name)
            .reset_index()
        )
```

生命周期如下：

1. `symbol_union()`从声明输入中取得当日股票并集；
2. `prepare_events()`完成过滤、订单匹配和事件整理，返回`symbol`及`time_ms`或
   `timestamp`；
3. 框架把事件时间转换为0至241的分钟编码，过滤分钟轴外记录；
4. `aggregate_minutes()`按股票和分钟返回稀疏结果；
5. 框架校验唯一性，补齐242分钟并转换为声明的值类型。

简单指标应覆写`aggregate_minutes()`并使用向量化`groupby`。复杂指标可只实现
`process_minute_data()`，但逐组Python调用通常更慢。需要共享聚合时，可调用
`finalize_aggregated_day()`把稀疏结果转换为标准日结果。

## 9. 运行单个指标

```python
from pathlib import Path

from trans_to_min import MinuteConversionRunner, RunConfig

config = RunConfig(
    input_root=Path("/sd1-data/level2"),
    output_root=Path(
        "/data/zhangyuan/trans_to_min/example_min/basic_min"
    ),
    date_workers=0,
    resume=True,
)

runner = MinuteConversionRunner(MinuteTradeCount(), config)
dates = runner.discover_dates("2025-01-02", "2025-01-31")
result = runner.run(dates)
```

`discover_dates()`负责发现并验证输入日期；`run()`只运行显式传入的日期列表。
`run([])`会报错，防止空任务被误判为成功。

仓库提供成交量参考入口：

```bash
cd /app/workspace/zhangyuan/Trans_To_Min

/app/workspace/zhangyuan/.venv/bin/python scripts/run_trade_volume.py \
  --start-date 2025-01-02 \
  --end-date 2025-01-31 \
  --input-root /sd1-data/level2 \
  --output-root /data/zhangyuan/trans_to_min/example_min/basic_min \
  --date-workers 0 \
  --arrow-threads-per-worker 4 \
  --resume
```

该脚本和`trade_volume_min`只是框架用法示例，不是正式业务数据定义。

## 10. 同源多指标组

`MinuteMetricGroup`适用于需要完全相同输入表集合的一组相关输出：

```text
InputBundle：每张物理表每个交易日读取一次
MetricGroup：共享输入列，可选择共享关联、分钟标记或聚合中间量
MetricOutput：每项正式数据独立目录、列、月份文件和manifest
```

```python
from pathlib import Path

from trans_to_min import MinuteMetricGroup, MinuteMetricGroupRunner

group = MinuteMetricGroup(
    "order_trade_group",
    (MetricA(), MetricB()),
)

runner = MinuteMetricGroupRunner(
    group,
    config,
    output_roots={
        "metric_a_min": Path("/data/topic/category_a"),
        "metric_b_min": Path("/data/topic/category_b"),
    },
)
runner.run(dates)
```

约束和语义：

- 组内指标必须使用完全相同的输入表集合，声明顺序可以不同；
- 实际读取列是各指标`input_columns`的并集；
- 默认只共享物理读取，之后仍逐指标调用原有`compute_day()`；
- 业务组可覆写`iter_compute_day()`共享更深层中间量；
- 覆写实现必须只产生所选指标，且每项恰好产生一次；
- 断点续跑按“指标×日期”判断，只计算尚未发布的输出；
- 多项指标按各自月份文件依次原子发布，整个指标组不是跨文件事务；
- 输入集合不同或业务上不应绑定运行的指标应使用不同组或单指标执行器。

每次只生成一种数据时，直接使用`MinuteConversionRunner`即可；指标组是可选的I/O和
中间计算复用能力，不是生产输出格式要求。

## 11. RunConfig完整说明

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `input_root` | `/sd1-data/level2` | 三类逐笔输入根目录 |
| `output_root` | `/data/zhangyuan/trans_to_min` | 单指标的类别目录或指标组共享暂存根目录 |
| `date_workers` | `0` | 日期进程数；`0`表示自动选择，正整数表示固定值 |
| `arrow_threads_per_worker` | `4` | 每个日期进程最多使用的PyArrow线程数 |
| `row_group_size` | `500000` | 正式及暂存Parquet的目标Row Group行数 |
| `compression` | `zstd` | 正式月份文件压缩算法 |
| `compression_level` | `3` | 正式月份文件压缩级别 |
| `staging_compression` | `snappy` | 单日暂存压缩；可设为`None`取消压缩 |
| `staging_compression_level` | `None` | 暂存压缩级别；Snappy和无压缩时禁止设置 |
| `auto_memory_fraction` | `0.70` | 自动并发最多使用的当前可用内存比例 |
| `auto_worker_memory_multiplier` | `16.0` | 输入未压缩列体积到单进程内存的初始放大倍数 |
| `adaptive_workers` | `True` | 是否按月用真实峰值校准后续并发 |
| `adaptive_memory_safety_factor` | `1.20` | 真实峰值校准的额外安全系数 |
| `publish_while_computing` | `True` | 是否让上月发布与下月计算重叠 |
| `overwrite_dates` | `False` | 是否重算并替换显式请求的已有日期 |
| `resume` | `False` | 是否跳过已有日期，只计算缺失日期 |

`overwrite_dates`和`resume`互斥。非法并发数、Row Group大小、内存比例或压缩组合会在
启动时立即报错。

## 12. 增量、覆盖和恢复语义

- 默认模式：目标月份存在请求日期时拒绝运行，防止隐式覆盖；
- `resume=True`：跳过已发布日期，只计算并插入缺失日期；
- `overwrite_dates=True`：重算指定日期，只替换这些日期，保留同月其他日期；
- 新月份：由当月日分片直接生成；
- 已有月份：顺序扫描旧文件一次，与新日分片按日期流式合并；
- 发布前校验Schema、行数、日期集合和顺序，然后使用`os.replace`原子替换；
- 运行失败时保留未完成暂存目录用于诊断；成功后自动清理本次暂存目录；
- 进程异常和月份发布异常会写入manifest并向调用方重新抛出，不静默跳过。

月文件按交易日升序保存。框架支持已有月份文件中跨交易日的Row Group，但不接受
交易日逆序或重复的日分片。

## 13. 并发和性能

框架按交易日使用多进程。每个月结束后销毁进程池，释放Pandas处理大表后可能长期
保留的内存。自动并发同时考虑：

- 操作系统可用CPU；
- `arrow_threads_per_worker`带来的总线程上限；
- 当前可用内存，包括Linux cgroup限制；
- 抽样日期所需Parquet列的未压缩体积；
- `auto_worker_memory_multiplier`和`auto_memory_fraction`。

自动模式会记录工作进程真实常驻内存峰值，并根据本月峰值、输入体积、下月体积和
安全系数调整下一月并发。显式`date_workers`始终优先，不做动态改写。

其他框架级优化：

- `time_ms`通过整数数组运算映射为`int16`分钟编码；
- 242个时间标签只保存一份分类字典；
- 稀疏聚合结果直接写入“分钟×股票”稠密数组；
- Arrow转Pandas后及时释放输入Arrow缓冲区；
- 日暂存默认使用Snappy，正式月份只执行一次ZSTD压缩；
- 上月流式发布默认与下月日期计算重叠。

这些优化只作用于读取、切片、补齐、调度和发布，不改变业务公式、股票集合、空值
语义、分钟边界或输出Schema。业务公式自身的向量化、NumPy归约或其他计算引擎选型
属于业务项目职责。

## 14. 公开接口索引

包根目录`trans_to_min`导出：

| 接口 | 用途 |
|---|---|
| `__version__` | 当前框架版本 |
| `RunConfig` | 不可变运行配置 |
| `L2InputAdapter` | 按列读取并归一化单日逐笔Parquet |
| `input_path` | 构造标准逐笔文件路径 |
| `MinuteMetric` | 单项分钟数据抽象基类 |
| `MinuteConversionRunner` | 单指标日期发现、计算和月份发布 |
| `MinuteMetricGroup` | 声明同源指标及输入列并集 |
| `MinuteMetricGroupRunner` | 共享读取、分项恢复和独立发布 |
| `ParallelismRecommendation` | 自动并发建议结果 |
| `recommend_date_workers` | 根据资源与输入元数据建议日期进程数 |

按模块使用的低层接口：

- `trans_to_min.adapter`：`discover_dataset_dates()`、`validate_matching_date_sets()`；
- `trans_to_min.minute`：`minute_labels()`、`assign_minute_labels()`、
  `assign_minute_codes()`、`assign_timestamp_minute_codes()`、
  `minute_labels_from_codes()`；
- `trans_to_min.resources`：`recommend_adaptive_date_workers()`、
  `process_peak_memory_bytes()`、`available_memory_bytes()`；
- `trans_to_min.storage`：`output_schema()`、`metric_root()`、`month_path()`、
  `published_dates()`、`write_day_frame()`、`publish_month()`、`write_manifest()`。

以下划线开头的函数和类是内部实现，不承诺跨版本稳定，业务项目不应直接依赖。

## 15. 测试与发布验收

运行完整框架测试：

```bash
python -m unittest discover -s tests -v
```

测试覆盖输入适配、缩放metadata、242分钟边界、股票并集、空分钟、可空整数、日期
完整性、月度流式增量、跨日期Row Group、指定日期覆盖、断点续跑、动态并发、禁止
隐式覆盖、同源输入共享和指标级恢复。

正式发布至少应完成：

1. `python -m compileall -q trans_to_min tests scripts`；
2. 完整单元测试通过；
3. `trans_to_min.__version__`、包元数据、README和变更记录一致；
4. 服务器目录文件与本地发布内容一致；
5. 服务器公共虚拟环境可导入框架；
6. Git提交和版本标签指向相同已验证内容；
7. 不把逐笔输入、分钟输出、暂存文件、日志或凭据提交到Git仓库。

业务项目还必须独立验证指标公式、方向口径、订单关联、分钟范围、空值、输出目录、
指标README和与外部基准的差异。框架测试通过不等于业务数据已经验收。

## 16. 版本与兼容性

本项目采用语义化版本号：

- 修订号变化用于兼容性修复；
- 次版本变化用于向后兼容的新能力；
- 主版本变化可能包含接口或固定口径调整。

`1.0.0`建立了242分钟轴、标准输出Schema、运行配置和发布语义的基础契约；
`1.0.1`将生产输入根目录固定为`/sd1-data/level2`。任何可能改变历史结果的修改，
都应提升版本、更新变更记录，并在正式数据README和manifest中保留所用框架版本。

GitHub仓库：<https://github.com/Fizzzy11/Trans_To_Min>
