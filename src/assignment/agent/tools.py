"""Tool definitions exposed to the model, in the OpenAI tool-calling format."""

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a bash command and return its stdout, stderr, and exit code. "
            "A non-zero exit code is reported, not raised.\n"
            "\n"
            "Every command runs in a new subshell, so a `cd` or an export does not "
            "carry over to the next command. Use the `cwd` and `env` arguments "
            "instead. Files you write do persist.\n"
            "\n"
            "Commands are non-interactive and cannot prompt for input, so pass "
            "flags like `-y` where a command would otherwise ask for confirmation. "
            "Prefer commands that produce little output; when reading a file, use "
            "`head`, `tail`, or `sed -n '10,20p'` rather than printing all of it.\n"
            "\n"
            "Useful patterns:\n"
            "- Create a file: `cat <<'EOF' > newfile.py` ... `EOF`\n"
            "- Edit in place: `sed -i 's/old/new/g' filename.py` (drop the trailing "
            "`g` to replace only the first match; restrict to a line range with "
            "`sed -i '1,10s/old/new/g'`)\n"
            "- View numbered lines: `nl -ba filename.py | sed -n '10,20p'`"
        ),
        # The nested env object intentionally accepts arbitrary variable names,
        # which is incompatible with strict schemas on some providers.
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": 'A shell command line, e.g. "ls -la | head".',
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                'The command as an argv list, e.g. ["ls", "-la"]. '
                                "Use this with shell=false when arguments contain "
                                "characters the shell would interpret."
                            ),
                        },
                    ],
                    "description": "The command to run.",
                },
                "shell": {
                    "type": ["boolean", "null"],
                    "description": (
                        "Whether to run the command through a shell, which enables "
                        "pipes, redirection, and globbing. Defaults to true. Set to "
                        "false when passing an argv list."
                    ),
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": (
                        "Absolute path to run the command in. Defaults to the "
                        "sandbox's current working directory."
                    ),
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": (
                        "Seconds to allow the command to run before killing it. "
                        "Defaults to no timeout."
                    ),
                },
                "env": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"},
                    "description": "Extra environment variables to set for this command.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": ("Send a message to the user."),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": ("Content of the message"),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}

INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "invoke_skill",
        "description": (
            "Load a skill and return its instructions. A skill is a short guide "
            "for one kind of work, written ahead of time.\n"
            "\n"
            "Call this before starting work a skill covers, and follow what it "
            "says in place of your default approach."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill's directory name, for example `hello-skill`."
                    ),
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.1.a): Define an OpenAI function-tool schema named ``play_move``.
# It must accept exactly one required string argument named ``move``, explain
# that moves use UCI notation (for example e2e4), and reject extra arguments.
PLAY_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "play_move",
        "description": ("Make a chess move using UCI notation. Examples: e2e4 (pawn move), e7e8q (pawn promotion to queen)."),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": ("UCI chess move string, e.g. e2e4, e7e8q."),
                },
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}


# TODO(3.3): Define the `simulate_move` tool, like the `play_move` tool.
# TODO(3.3.a): Define an OpenAI function-tool schema named ``simulate_move``.
# Accepts required fen (full six-field FEN), optional move (UCI string).
# If move omitted: returns position + all legal moves.
# If fen + move provided: simulate exactly one ply, return resulting board state.
# This tool does NOT modify the live game state, only simulates.
SIMULATE_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "simulate_move",
        "description": (
            "Simulate a chess board position or preview a chess move using UCI notation. "
            "When only a full six-field FEN is supplied: returns the board position and all legal moves for that position. "
            "When both FEN and UCI move are supplied: executes exactly one ply and returns the resulting board state. "
            "This is a read-only simulation tool and will not alter the active game. "
            "Examples: fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', move='e2e4'"
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "Complete six-field FEN string representing a chess board position.",
                },
                "move": {
                    "type": "string",
                    "description": "Optional UCI notation move string, e.g. e2e4, e7e8q. Omit this argument to inspect the position and its legal moves.",
                },
            },
            "required": ["fen"],
            "additionalProperties": False,
        },
    },
}


# TODO()
# TODO(3.4): Define OpenAI function-tool schema for run_python
RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a snippet of Python code inside a sandbox. "
            "The sandbox has access to simulate_move and play_move functions. "
            "Use this to search, look ahead, and select chess moves programmatically."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code string to execute inside sandbox."
                }
            },
            "required": ["code"],
            "additionalProperties": False
        }
    }
}

