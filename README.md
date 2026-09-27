# SageMatch（知弈）

岗位定制化模拟面试训练系统。项目名仅作代号，界面与文案一律使用「模拟面试」。

第一期：像素还原会话舱 + 模拟面试（入口 / 进行中 / 复盘），后端跑通「岗位提交 → 出题 → 文字面试 → 复盘」。管理端先做可进的壳。语音 / RAG 评测后置。

## 目录结构

- `backend/app/core/`：配置、数据库与运行策略；`integrations/`、`materials/` 放模型和材料底层适配。
- `backend/app/agents/`：`contracts/` 定义状态与协议，`roles/` 放角色逻辑，`providers/` 管理模型路由，`tools/` 管理工具，`orchestration/` 管理总图、checkpoint 与追踪，`workflows/` 放业务子图。
- `backend/app/api/` 与 `services/`：按聊天、面试、材料、管理/运行职责分包；`models/` 与 `schemas/` 分业务实体和平台/管理契约。
- `backend/tests/unit/` 按业务边界组织快速测试，`integration/postgres/` 验证隔离数据库中的并发与恢复，`fixtures/` 存合成数据。
- `frontend/src/api/` 分 HTTP 核心与领域接口；`features/` 放业务页面及私有组件；`components/`、`layout/`、`lib/` 放共享 UI、外壳和通用能力。

## 本地启动

需要：Python 3.11、Node 24、本机 PostgreSQL（库名 `sagematch`）。复制 `.env.example` 为 `.env` 并填密码与 LLM 密钥。

```powershell
# 后端
python -m venv .venv
cd backend
..\.venv\Scripts\python -m pip install -r requirements.txt
# Psycopg 异步 checkpoint 在 Windows 上需要 Selector 事件循环。
..\.venv\Scripts\python run.py --reload --port 8000

# 前端（另开一个终端）
cd frontend
npm install
npm run dev
```

- 用户端：http://localhost:5173
- 管理端：http://localhost:5173/admin
- API 文档：http://127.0.0.1:8000/docs

单人私有使用且需要在启动前强制核对真实 RAG 门禁时，从仓库根目录运行
`pwsh -NoProfile -File .\backend\scripts\start_private_checked.ps1`。该入口先核对
当前 `ready` 索引与私有 `data/eval` 的固定集、原始报告和人工复核结果，再将后端
绑定到 `127.0.0.1:8000`；门禁失败时不会启动后端。开发时可继续使用上面的
`run.py --reload`，材料索引变更后须重新评测并复核门禁产物。

后端回归从仓库根目录执行 `.\.venv\Scripts\python.exe -m pytest backend/tests -q`；前端运行 `npm run build` 和 `node --test tests/http-sse.test.mjs`。
