"""Falcon's assistant: a language-model agent that uses the app through typed tools (docs/assistant.md).

tool_api: what a tool is, the registry, the tiers and what is never allowed. Every action the assistant can take is a
Tool calling the same code the GUI and the CLI call; nothing reaches the app any other way.
"""
