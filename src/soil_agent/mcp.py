"""Minimal MCP tools/list and tools/call bridge for the Agent tool layer."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Protocol

from .tools import ToolRegistry, ToolResult


class MCPTransport(Protocol):
    def __call__(self, message: dict[str, Any]) -> dict[str, Any]: ...


class MCPToolServer:
    """Expose a ToolRegistry through the MCP JSON-RPC method shape."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        method = message.get("method")
        request_id = message.get("id")
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {"name": name, "description": tool.__class__.__doc__ or name, "inputSchema": {"type": "object"}}
                        for name, tool in self.registry.tools.items()
                    ]
                },
            }
        if method == "tools/call":
            params = message.get("params", {})
            try:
                result = self.registry.invoke(str(params["name"]), dict(params.get("arguments", {})))
                return {"jsonrpc": "2.0", "id": request_id, "result": {"isError": not result.ok, "structuredContent": asdict(result)}}
            except Exception as exc:
                return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}}


class MCPToolClient:
    def __init__(self, transport: MCPTransport):
        self.transport = transport

    def list_tools(self) -> list[dict[str, Any]]:
        response = self.transport({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        return list(response.get("result", {}).get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        response = self.transport({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        if "error" in response:
            return ToolResult(name, False, error=str(response["error"].get("message", "MCP tool error")))
        structured = response.get("result", {}).get("structuredContent", {})
        return ToolResult(
            tool=str(structured.get("tool", name)),
            ok=not bool(response.get("result", {}).get("isError", False)),
            result_ids=tuple(structured.get("result_ids", ())),
            evidence=tuple(structured.get("evidence", ())),
            data=structured.get("data"),
            error=str(structured.get("error", "")),
        )


class MCPProxyTool:
    def __init__(self, client: MCPToolClient, name: str):
        self.client = client
        self.name = name

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        return self.client.call_tool(self.name, arguments)
