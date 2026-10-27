"""Model access. Everything goes through Microsoft Agent Framework chat clients, configured by environment only.

Preferred (Microsoft Foundry project, Entra ID auth - no API key needed, works with a managed identity):
    FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
    FOUNDRY_MODEL=<model deployment name>

Alternative (Azure OpenAI resource):
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
    AZURE_OPENAI_DEPLOYMENT=<deployment name>
    AZURE_OPENAI_API_KEY=<key>            # optional; without it DefaultAzureCredential is used
    AZURE_OPENAI_API_VERSION=<optional>

INSIGHT_LLM=off forces the deterministic template path (nothing else changes).
"""
import os


def make_chat_client():
    """-> (client | None, label). `label` is the model name for provenance, or the reason no client was built."""
    if os.getenv("INSIGHT_LLM", "auto").strip().lower() in ("off", "0", "false", "no"):
        return None, "disabled (INSIGHT_LLM=off)"
    try:
        ep, model = os.getenv("FOUNDRY_PROJECT_ENDPOINT"), os.getenv("FOUNDRY_MODEL")
        if ep and model:
            from agent_framework.foundry import FoundryChatClient
            from azure.identity import DefaultAzureCredential
            return FoundryChatClient(project_endpoint=ep, model=model, credential=DefaultAzureCredential()), f"foundry:{model}"
        aep, dep = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if aep and dep:
            from agent_framework.openai import OpenAIChatCompletionClient
            key = os.getenv("AZURE_OPENAI_API_KEY")
            kw = {"api_key": key} if key else {}
            if not key:
                from azure.identity import DefaultAzureCredential
                kw["credential"] = DefaultAzureCredential()
            if os.getenv("AZURE_OPENAI_API_VERSION"):
                kw["api_version"] = os.environ["AZURE_OPENAI_API_VERSION"]
            return OpenAIChatCompletionClient(model=dep, azure_endpoint=aep, **kw), f"azure-openai:{dep}"
    except Exception as ex:  # a broken config must degrade to templates, never crash the API
        return None, f"client init failed: {type(ex).__name__}: {ex}"
    return None, "not configured (set FOUNDRY_PROJECT_ENDPOINT + FOUNDRY_MODEL)"
