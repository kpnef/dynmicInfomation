# HIEVE Track1 模拟采样 + ByteTrack 预测补全 + MOTP/IDF1 评估

本项目实现你描述的 1~5 流程（并按你最新要求：**GT + 公共探测(det) + 每路独立 FPS，通过 JSON 配置**）：

1. **读取 HIEVE Track1 (MOT 格式) 标注(GT)**，默认每个视频起点时间 = 0。
2. **融合 calc.py 风格的计算框架**，用“虚拟时间轴(ms)”驱动多通道调度：每个 tick 只处理一个通道（等权），从而得到**稀疏帧**的检测输入。
3. 稀疏输入下，调用 **ByteTrack.update(..., n=gap_frames)** 进行多步 KF 预测，并通过 `pop_box_history()` 拿到**每一步预测框**，从而补全为**全帧输出**（与标注帧数等长）。
4. 对“预测输出 vs 标注”做 **MOTP + IDF1**（轻量实现，Hungarian + IoU 阈值）。
5. ByteTrack Update 被直接引用；calc 引擎已融合成项目代码的一部分。

## 目录结构

- `hieve_sim/`
  - `hieve_reader.py` 读取 MOT 标注(GT)与 MOT 检测(det)
  - `config.py` JSON 配置解析
  - `calc_engine.py` 计算/调度框架（虚拟时间）
  - `track/` ByteTrack（去掉 ultralytics 依赖，保留核心逻辑），`update()` + `pop_box_history()`
  - `simulator.py` 主流程：稀疏采样 → ByteTrack 预测补全 → 保存/评估
  - `metrics.py` MOTP / IDF1
  - `cli.py` 命令行入口（支持 `--config`）
- `run_hieve_sim.py` 直接运行脚本

## JSON 配置（推荐）

配置文件是一个 JSON 对象，核心字段：

```json
{
  "tick_ms": 300,
  "iou_thr": 0.5,
  "det_score_thr": 0.1,
  "tracker_cfg": {
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "track_buffer": 90,
    "match_thresh": 0.8,
    "fuse_score": false
  },
  "sources": [
    {
      "name": "cam0",
      "gt": "relative/path/to/gt.txt",
      "det": "relative/path/to/det.txt",
      "fps": 30
    }
  ]
}
```

说明：
- `sources[]`：数组元素对应一个“真实视频源”
  - `gt`：相对路径（相对 **config.json 所在目录**）
  - `det`：相对路径（相对 **config.json 所在目录**）
  - `fps`：该路数据帧率（用于虚拟时间→帧号映射；每路可不同）
- `det_score_thr`：可选；过滤探测分数低的框（很多公共探测文件会很密）
- `tracker_cfg`：可选；覆盖 ByteTrack 参数

### 支持“空通道/空数据”模拟（按你的最新定义）

你定义的规则是：**只要该通道的 `gt` 或 `det` 任何一个是 `null`/缺失，就把这个通道当作从头到尾“完全没有目标”**。

因此本项目在解析 JSON 时会强制：

- 若 `gt` 为 `null`/缺失 **或** `det` 为 `null`/缺失：则自动设置 `gt=null` 且 `det=null`，并强制 `num_frames=0`。

效果：该通道被视为**没有任何帧**，因此会被调度器与 ByteTrack **完全跳过**（不计算，不影响 overall 指标）。

此外，程序会输出：
- **每个通道的 MOTP/IDF1**
- **整体（所有通道汇总）的 MOTP/IDF1**

运行：

```bash
python run_hieve_sim.py --config demo_config.json --save-pred --out-dir outputs
```

> 说明：为满足“det 必须来自 det 文件”的项目约束，旧的 `--labels`（GT 当 det）模式已禁用。


## Empty channels (compute dilution experiment)

Two different concepts are supported:

1) **Fully-empty camera (SKIPPED)**: if either `gt` or `det` is null/missing for a source, the source is treated as inactive and is *not scheduled*.
2) **Empty stream (SCHEDULED)**: a source with `"empty": true` (or `--empty-channels N`) participates in scheduling but has no objects (GT/DET are empty lists on all frames).

### Configure via JSON
- Root field `empty_channels`: auto-insert N scheduled empty streams.
- Or per-source: `{ "name": "empty0", "fps": 30, "empty": true, "num_frames": 2000 }`

### Configure via CLI
- `--empty-channels N` overrides `empty_channels` from config.

Example:
```bash
python run_hieve_sim.py --config your_config.json --empty-channels 4
```
