"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, INVOKE_SKILL_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment


def _tool_message(tool_call_id: str, content: str) -> dict[str, str]:
    """One observation, linked to the call it answers."""

    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.tools.append(EXECUTE_TOOL)
        self.tools.append(SEND_MESSAGE_TOOL)

        self.task = task
        self.submitted_patch = ""

        system_information = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )
        self.system_prompt = (
            "You are a software engineering agent working in a Linux container, "
            "on a repository checked out at /testbed.\n"
            "\n"
            "You are given a bug report. Reproduce the failure, find its cause, "
            "fix it in the repository, and verify the fix by running the "
            "relevant tests.\n"
            "\n"
            "Every command runs in a fresh shell, so a `cd` or an export does "
            "not carry over to the next call. Prefer narrow commands (`sed -n`, "
            "`grep -n`, `tail`) over printing whole files.\n"
            "\n"
            "<system_information>\n"
            f"{system_information}\n"
            "</system_information>\n"
        )
        self.task_prompt = (
            "Fix the following issue in the repository at /testbed.\n\n" + self.task
        )

        # Progressive disclosure: the catalog carries each skill's name and
        # description only. A skill's body arrives when the agent invokes it, so
        # an agent with no skills is never told how to submit.
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute the calls this agent recognizes and observe each one."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations: list[dict[str, str]] = []

        for call in tool_calls:
            call_id = (
                call.get("id", "unknown_id") if isinstance(call, dict) else "unknown_id"
            )

            # Only parsing is wrapped. A call we cannot read is the model's
            # mistake, so it comes back as an observation to correct. A sandbox
            # that has died is terminal instead, so `Environment.execute` is
            # called outside this `try` and its RuntimeError propagates rather
            # than being relayed as though the model had written bad arguments.
            try:
                function = call["function"]
                name = function["name"]
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    raise ValueError("tool arguments must be a JSON string")
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must decode to a JSON object")
            except Exception as exc:
                observations.append(
                    _tool_message(
                        call_id, f"Error: could not read this call's arguments: {exc}"
                    )
                )
                continue

            if name == EXECUTE_TOOL["function"]["name"]:
                command = parsed.get("command")
                if command is None:
                    observations.append(
                        _tool_message(call_id, "Error: `execute` requires a `command`.")
                    )
                    continue
                result = self.env.execute(
                    command=command,
                    shell=parsed.get("shell"),
                    cwd=parsed.get("cwd"),
                    timeout=parsed.get("timeout"),
                    env=parsed.get("env"),
                )
                content = format_tool_output(result)
            elif name == SEND_MESSAGE_TOOL["function"]["name"]:
                summary = parsed.get("summary")
                if not isinstance(summary, str):
                    observations.append(
                        _tool_message(
                            call_id, "Error: `send_message` requires a `summary`."
                        )
                    )
                    continue
                self.submitted_patch = summary
                self.finished = True
                content = f"Message submitted:\n{summary}"
            elif name == INVOKE_SKILL_TOOL["function"]["name"]:
                skill_name = parsed.get("name")
                if not isinstance(skill_name, str):
                    observations.append(
                        _tool_message(
                            call_id, "Error: `invoke_skill` requires a `name`."
                        )
                    )
                    continue
                skill = self.skills.get(skill_name)
                if skill is None:
                    available = ", ".join(sorted(self.skills)) or "none"
                    observations.append(
                        _tool_message(
                            call_id,
                            f"Error: no skill named `{skill_name}`. "
                            f"Available skills: {available}.",
                        )
                    )
                    continue
                content = skill["content"]
            else:
                available = ", ".join(tool["function"]["name"] for tool in self.tools)
                content = f"Error: unknown tool `{name}`. Available tools: {available}."

            observations.append(_tool_message(call_id, content))

        return observations
