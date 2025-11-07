# MCP Integration Guide

This guide explains how to connect your IDE to a Model Context Protocol (MCP) server so the `rag-demo` agent can enrich its answers with live weather data. The steps use the community Weather MCP server as an example, but the configuration supports any MCP-compliant provider.

## 1. Install the MCP client tooling

Most modern IDEs (Cursor, VS Code MCP extension, JetBrains AI) support MCP by loading a `mcp.config.json` file from your workspace. Ensure you are running a recent build of the IDE and enable MCP support if it is behind a feature flag.

```bash
# Optional: confirm the weather MCP package is available
npx -y @modelcontextprotocol/weather-mcp --help
```

## 2. Provide an API key

The Weather MCP uses the OpenWeather API. Create a free account at <https://openweather.co.uk> and export the key before launching your IDE:

```bash
export OPENWEATHER_API_KEY="sk-..."
```

> **Tip:** add the variable to your shell profile (e.g., `~/.bashrc`) so the IDE inherits it automatically.

## 3. Review `mcp.config.json`

The repository now ships with `mcp.config.json` in the project root:

```json
{
  "version": 1,
  "servers": {
    "weather-mcp": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/weather-mcp"],
      "workingDirectory": ".",
      "env": {
        "OPENWEATHER_API_KEY": "${OPENWEATHER_API_KEY}"
      }
    }
  },
  "prompts": {
    "enrich-with-weather": {
      "description": "Pull in real-time weather context to augment RAG answers.",
      "mcpServers": ["weather-mcp"]
    }
  }
}
```

Place the file in the project root (already committed) so IDEs can discover it automatically. When the IDE starts it will spawn the Weather MCP server by running `npx -y @modelcontextprotocol/weather-mcp` with the environment variables populated from your session.

## 4. Use the enrichment prompt

The configuration defines a reusable prompt called `enrich-with-weather`. In agents that support MCP prompts (including Cursor MCP chat and LangChain MCP bindings), call the prompt before handing back to the main RAG flow:

```python
from langchain.schema import HumanMessage
from langchain_core.messages import SystemMessage

# Pseudocode: ask the MCP prompt for weather context
weather_context = mcp_client.invoke_prompt(
    "enrich-with-weather",
    messages=[HumanMessage(content="What is the weather in London today?")],
+)

# Inject into your final answer
answer = f"Weather snapshot:\n{weather_context}\n\n" + rag_agent_answer
```

This approach follows the guardrail and routing guidelines recommended in the [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview): we add a dedicated router/guard component in front of the main agent and provide a specialised tool (the MCP server) for enrichment.

## 5. Troubleshooting

| Issue | Fix |
| --- | --- |
| MCP server fails to start | Confirm `npx` is installed (Node.js ≥ 18) and `OPENWEATHER_API_KEY` is exported. |
| IDE cannot find configuration | Ensure `mcp.config.json` lives at the repository root or update the IDE setting to point to it. |
| Rate limits from OpenWeather | Switch to the paid tier or lower the call frequency; the demo only needs current conditions. |

With this configuration, every IDE session can enrich RAG responses with real-time weather data while keeping the main application code unchanged.
