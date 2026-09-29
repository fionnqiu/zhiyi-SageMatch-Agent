"""Availability must follow the same provider routing as model calls."""

from types import SimpleNamespace
from unittest.mock import patch

from app.services.operations.llm_gateway import provider_available


def test_database_provider_enables_role_without_environment_key() -> None:
    choice = SimpleNamespace(
        provider=SimpleNamespace(api_key="database-key"),
        binding=SimpleNamespace(model="bound-model"),
    )
    settings = SimpleNamespace(llm_api_key="", llm_model="")
    with patch("app.services.operations.llm_gateway.choose_provider", return_value=choice), patch(
        "app.services.operations.llm_gateway.get_settings", return_value=settings
    ):
        assert provider_available(object(), "author")


def test_unconfigured_role_is_unavailable() -> None:
    choice = SimpleNamespace(provider=None, binding=None)
    settings = SimpleNamespace(llm_api_key="", llm_model="default-model")
    with patch("app.services.operations.llm_gateway.choose_provider", return_value=choice), patch(
        "app.services.operations.llm_gateway.get_settings", return_value=settings
    ):
        assert not provider_available(object(), "critic")
