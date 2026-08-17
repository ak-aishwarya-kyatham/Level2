import importlib
import pytest

@pytest.mark.unit
def test_backend_modules_import_smoke():
    """
    Lightweight smoke test to verify all core backend modules can be imported
    without raising NameError, ImportError, SyntaxError, or missing dependency errors.
    """
    modules_to_check = [
        "app.agents.policy_agent",
        "app.agents.reflection_agent",
        "app.agents.response",
        "app.workflows.main_workflow",
        "app.mcp_client",
        "app.mcp_server",
        "app.utils.evaluator",
        "app.agents.ingestion",
        "app.agents.search_retrieval",
        "app.agents.cleaning",
        "app.agents.triage",
        "app.repositories.news_repository",
        "app.routers.chat",
        "app.routers.analytics"
    ]

    for mod_name in modules_to_check:
        # Do not wrap in try-except block so any import/name error immediately fails pytest
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Module {mod_name} imported as None"
