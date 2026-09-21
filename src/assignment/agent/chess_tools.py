"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.
    try:
        if not isinstance(arguments, str):
            return "<chess_error>arguments must be a JSON string</chess_error>"

        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")

        fen = parsed.get("fen")
        if not isinstance(fen, str) or not fen.strip():
            raise ValueError("fen must be a non-empty six-field FEN string")

        move = parsed.get("move")
        # Allow move=None (from sandbox move:null), reject non-string non-None values
        if move is not None:
            if not isinstance(move, str) or not move.strip():
                raise ValueError("move must be a UCI notation string or omitted")

        payload = {"fen": fen}
        if move is not None:
            payload["move"] = move

        state = _request_state(client, "POST", "/api/simulate", json=payload)
        return json.dumps(state)

    except json.JSONDecodeError as e:
        return f"<chess_error>invalid JSON arguments: {e}</chess_error>"
    except httpx.TransportError as e:
        return f"<chess_error>transport failure contacting chess server: {e}</chess_error>"
    except ValueError as e:
        return f"<chess_error>invalid simulate_move arguments: {e}</chess_error>"
    except RuntimeError as e:
        return f"<chess_error>simulate_move server error: {e}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    # TODO(3.1.b): Parse the arguments and POST {"move": <uci move>} to
    # /api/move. Return its JSON object. Catch any errors raised by the
    # tool and return an error message between `<chess_error></chess_error>`
    # for the agent to address. Cover malformed JSON arguments, arguments
    # that are not an object, a missing or non-string fen, a non-string move,
    # a position or move the server rejects, and a transport failure.
    try:
        # 新增：arguments类型校验，拦截 None / dict / int 等非字符串，避免json.loads抛TypeError
        if not isinstance(arguments, str):
            return "<chess_error>Tool arguments must be a JSON string.</chess_error>"

        # 解析原始JSON字符串
        args = json.loads(arguments)
        # 校验必须是JSON对象dict
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        # 校验move字段存在
        if "move" not in args:
            raise ValueError("Missing required argument: move")
        move_val = args["move"]
        # 校验move是非空UCI字符串
        if not isinstance(move_val, str) or not move_val.strip():
            raise ValueError("move must be a non-empty UCI string.")

        # POST 请求到 /api/move
        state_dict = _request_state(client, "POST", "/api/move", json={"move": move_val})
        # 成功：序列化state字典为json字符串返回
        return json.dumps(state_dict)

    # 顺序不能变！JSONDecodeError是ValueError子类，放前面
    except json.JSONDecodeError as e:
        return f"<chess_error>Malformed JSON arguments: {e}</chess_error>"
    except httpx.TransportError as e:
        return f"<chess_error>Transport failure: {e}</chess_error>"
    except ValueError as e:
        return f"<chess_error>Bad arguments or server rejected move: {e}</chess_error>"
    except RuntimeError as e:
        return f"<chess_error>Invalid server response: {e}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    #
    # `/opt/assignment/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    raise NotImplementedError


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    # TODO(3.5): parse the arguments and return the named skill's content.
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    raise NotImplementedError


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
