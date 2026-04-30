import ast

BANNED_IMPORTS = {"os", "subprocess", "sys", "shutil", "socket", "requests", "urllib", "http"}

_ALWAYS_REQUIRED = {"detect_rep"}
_MODE_FUNCTIONS = {
    "during_exercise": "generate_rep_cue",
    "after_window": "generate_round_feedback",
    "after_rep": "generate_rep_feedback",
    "after_exercise": "get_session_summary",
}


def validate_module(code: str, feedback_mode: list[str] | None = None) -> dict:
    """
    Validate a generated analysis module.
    Returns {"valid": bool, "error": str | None}.
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {e}"}

    # 2. Check banned imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BANNED_IMPORTS:
                    return {"valid": False, "error": f"Banned import: '{top}'"}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BANNED_IMPORTS:
                    return {"valid": False, "error": f"Banned import: '{top}'"}

    # 3. Check required functions are defined at module level
    modes = set(feedback_mode) if feedback_mode else {"after_window"}
    required = _ALWAYS_REQUIRED | {_MODE_FUNCTIONS[m] for m in modes if m in _MODE_FUNCTIONS}
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = required - defined
    if missing:
        return {"valid": False, "error": f"Missing required functions: {', '.join(sorted(missing))}"}

    return {"valid": True, "error": None}
