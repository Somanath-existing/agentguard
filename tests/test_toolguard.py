"""
tests/test_toolguard.py
Tests for ToolGuard — allowlist, denylist, argument injection detection, and custom validators.
"""
from __future__ import annotations

from agentguard.toolguard import ToolGuard


def test_permissive_by_default():
    guard = ToolGuard()
    result = guard.authorize("any_tool")
    assert result.allowed is True
    assert result.reason == "Authorized"


def test_allowlist():
    guard = ToolGuard(allowlist=["search", "calc_*"])
    
    # Matching tool allowed
    res1 = guard.authorize("search")
    assert res1.allowed is True
    
    # Matching pattern allowed
    res2 = guard.authorize("calc_add")
    assert res2.allowed is True
    
    # Non-matching tool denied
    res3 = guard.authorize("shell")
    assert res3.allowed is False
    assert "not in allowlist" in res3.reason


def test_denylist():
    guard = ToolGuard(denylist=["exec_*", "bash"])
    
    # Non-matching allowed
    res1 = guard.authorize("search")
    assert res1.allowed is True
    
    # Matching pattern denied
    res2 = guard.authorize("exec_code")
    assert res2.allowed is False
    assert "matches denylist pattern" in res2.reason
    
    # Matching exact tool denied
    res3 = guard.authorize("bash")
    assert res3.allowed is False


def test_allowlist_takes_precedence():
    # If both are specified, allowlist should be used, and denylist is ignored
    guard = ToolGuard(allowlist=["search"], denylist=["search"])
    res = guard.authorize("search")
    assert res.allowed is True


def test_argument_injection():
    guard = ToolGuard(arg_inspection=True)
    
    # Safe args allowed
    res1 = guard.authorize("run_cmd", "echo 'hello'")
    assert res1.allowed is True
    
    # Dangerous args denied (semicolon command chaining)
    res2 = guard.authorize("run_cmd", "cat file.txt; rm -rf /")
    assert res2.allowed is False
    assert "Dangerous argument pattern" in res2.reason
    
    # Dangerous args denied (backticks)
    res3 = guard.authorize("run_cmd", "echo `whoami`")
    assert res3.allowed is False
    
    # Dangerous args denied (dollar paren)
    res4 = guard.authorize("run_cmd", "echo $(whoami)")
    assert res4.allowed is False


def test_custom_validator():
    def my_validator(tool_name: str, tool_args: str) -> str | None:
        if "bad_arg" in tool_args:
            return "Block due to bad_arg"
        return None
        
    guard = ToolGuard(custom_validators=[my_validator])
    
    res1 = guard.authorize("tool", "good_arg")
    assert res1.allowed is True
    
    res2 = guard.authorize("tool", "some bad_arg here")
    assert res2.allowed is False
    assert res2.reason == "Block due to bad_arg"
