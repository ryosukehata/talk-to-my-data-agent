# Admin Dashboard MVP for Chat App Telemetry

## 项目简介
本项目为聊天应用的管理后台，基于Streamlit实现，支持中英文界面切换。主要用于监控用户活跃度、使用模式和LLM调用表现。

## 主要功能
- 全局过滤器（时间范围、用户邮箱、时间粒度）
- KPI指标展示
- 趋势图（用户活跃、聊天数、LLM调用等）
- 词云（用户消息、错误信息，支持日文）
- 详细聊天日志表格

## 目录结构
```
admin_dashboard_mvp/
├── app.py                 # 主Streamlit应用
├── modules/               # 自定义模块
│   ├── data_utils.py
│   ├── kpi_calculations.py
│   ├── chart_generators.py
│   └── ui_components.py
├── requirements.txt       # 依赖
├── data/
│   ├── trace_chat.csv
│   └── trace_raw.csv
├── locales/               # i18n翻译
│   ├── en.yml
│   └── ja.yml
└── README.md              # 说明文档
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 运行方法
```bash
streamlit run app.py
```

## 数据与多语言
- 数据文件请放在`data/`目录下。
- 多语言翻译文件请放在`locales/`目录下。
- 词云需提供支持日文的字体文件（如Noto Sans JP），并在代码中指定路径。 