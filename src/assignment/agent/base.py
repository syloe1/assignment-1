"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
COMPACTION_SYSTEM_PROMPT = """You compress one software agent's ReAct history \
into a short factual working memory.

Write plain prose. No markdown, no headings, no preamble.

Keep, when the history supports it:
- the objective and any constraints stated for it
- files read or changed, and the edits made to them
- commands run and their concrete results: exit codes, test outcomes, error text
- approaches that failed, and why they failed
- what is blocked, and the next action to take

Drop:
- output that was superseded or repeated, such as the same file read twice
- raw file dumps, long logs, and detail no later step relied on

Record only what the history states. Do not invent results, and do not add
instructions, advice, or next steps of your own."""


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


def parse_yaml_frontmatter(text: str, source: str | Path) -> dict[str, Any]:
    """Parse the YAML mapping between the leading `---` markers of a skill file.

    Args:
        text: The whole file, as read from disk.
        source: Where it came from, named in any error raised.

    Raises:
        ValueError: If the markers are missing or unterminated, or the block
            between them is not valid YAML mapping syntax.
    """

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source} does not start with a `---` frontmatter block")

    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing = index
            break
    if closing is None:
        raise ValueError(f"{source} has an unterminated `---` frontmatter block")

    try:
        parsed = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} has malformed YAML frontmatter: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{source} frontmatter is not a YAML mapping")
    return parsed


def group_history_into_steps(
    history: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group history into assistant actions, each with the observations answering it.

    A step starts at an assistant message and runs up to the next one, so an
    action that made several parallel calls stays whole. Compaction has to cut
    between steps and never inside one: a tool observation separated from the
    call it answers is a message sequence the API rejects.
    """

    steps: list[list[dict[str, Any]]] = []
    for message in history:
        if message.get("role") == "assistant" or not steps:
            steps.append([message])
        else:
            steps[-1].append(message)
    return steps


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self,
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
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # Every assistant action and the tool observations answering it, in
        # the order the model will see them. The system and task messages are
        # deliberately NOT stored here: `build_prompt` assembles those from
        # `system_prompt` / `task_prompt` on every call, so a subclass stays
        # in control of its own opening messages.
        self.history: list[dict[str, Any]] = []

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent."""

        # TODO(1.4): Validate ``skills_path``, discover one ``SKILL.md``
        # per child directory, parse its YAML frontmatter (what's between the
        # `---` tags at the head of the file), and return a mapping
        # keyed by the frontmatter ``name``. Each value must contain a concise
        # ``metadata`` string for the model's skill catalog and the complete
        # ``content`` of the skill file for ``invoke_skill``. Reject duplicate
        # names and malformed or missing frontmatter with a clear
        # ``ValueError``.
        if not skills_path.is_dir():
            raise ValueError(f"skills_path is not a directory: {skills_path}")

        skills: dict[str, dict[str, str]] = {}
        for directory in sorted(skills_path.iterdir()):
            if not directory.is_dir():
                continue

            skill_file = directory / "SKILL.md"
            if not skill_file.is_file():
                raise ValueError(f"Skill directory {directory.name} has no SKILL.md")

            content = skill_file.read_text(encoding="utf-8")
            frontmatter = parse_yaml_frontmatter(content, skill_file)
            name = frontmatter.get("name")
            description = frontmatter.get("description")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{skill_file} has no `name` in its frontmatter")
            if not isinstance(description, str) or not description:
                raise ValueError(
                    f"{skill_file} has no `description` in its frontmatter"
                )
            if name in skills:
                raise ValueError(f"Duplicate skill name `{name}` in {skill_file}")

            # The catalog carries only the frontmatter, so the skill's body stays
            # out of the prompt until the agent calls `invoke_skill` for it.
            skills[name] = {
                "metadata": f"name: {name}\ndescription: {description}",
                "content": content,
            }
        return skills

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response."""

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        """Assemble the message sequence sent to the language model.

        The opening messages come from the subclass, so this method never has
        to know which domain it is serving. Everything after them is the
        accumulated history, appended by the loop as the run proceeds.

        This is a pure read: it is called several times per step (by
        `query_language_model`, `estimate_active_prompt_tokens`, and twice by
        `maybe_compact_context`), so it must never modify agent state.

        Returns:
            The standing instructions, the task statement, then every action
            and observation so far, in the order the API expects.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        ]
        messages.extend(self.history)
        return messages

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage."""

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all."""

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits."""
        # The opening system/task messages are not in `history`, so summarizing
        # a prefix of it keeps them verbatim for free.
        steps = group_history_into_steps(self.history)
        keep_recent = self.compaction_keep_recent_steps
        if len(steps) <= keep_recent:
            return [], {}
        old_steps = steps[:-keep_recent]
        kept_steps = steps[-keep_recent:]

        # TODO(2.1): Prompt the model to compact the context. The system
        # prompt should ask for concise factual working memory and preserve
        # the objective, constraints, files, commands, edits, concrete
        # results, failed approaches, tests, blockers, and next action.
        # Summarize only an old prefix; retain the original system/task
        # messages verbatim and at least the latest complete assistant action
        # with all linked tool observations. The resulting summary should change
        # what `build_prompt` emits, and reduce the length of the prompt.
        compaction_prompt = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{self.task_prompt}\n\n"
                    "Work performed so far, to compress:\n"
                    f"{json.dumps(old_steps, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.
        summary = compaction_response.choices[0].message.content or ""
        self.history = [
            # A `user` message, because the step after it opens with an assistant
            # message: two assistant messages in a row is not a sequence the API
            # accepts, and this is information handed to the model, not a tool
            # observation. The tag keeps it from reading as a new instruction.
            {
                "role": "user",
                "content": f"<working_memory>\n{summary}\n</working_memory>",
            }
        ] + [message for step in kept_steps for message in step]

        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached."""

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            # TODO(1.2) Run the ReAct loop. Orchestrate the sequence of
            # prompting the language model to produce reasoning and actions,
            # extracting the tool calls produced by the model, and executing
            # the tool calls to obtain the agent's observation for the next
            # step. Ensure you identify when the agent has completed the task
            # by setting `Agent.finished`. If the agent exceeds the
            # `step_limit`, raise `StepLimitError`.

            # React 主循环
            while not self.finished:
                # Budget first: there is no next action request to compact for
                # once the limit is reached, and compacting anyway would spend a
                # model call on a step that never runs.
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(f"Reached step limit {self.step_limit}")

                # 压缩上下文
                self.maybe_compact_context()

                # 1. Prompting LLM
                assistant_msg =  self.query_language_model()

                # 追加进入history
                self.history.append(assistant_msg)

                # extracting tool_calls from model response 
                tool_calls = assistant_msg.get("tool_calls", [])

                if not tool_calls:
                    #"response contained no parsed tool call; the loop should preserve the response and continue"
                    continue 

                # executing tool calls 
                observations = self.execute_tool_calls(tool_calls)
                # 把observation 写入history 多个观察用extend
                self.history.extend(observations)

        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        raise NotImplementedError
