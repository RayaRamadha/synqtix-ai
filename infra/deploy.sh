#!/usr/bin/env bash
# Deploy to Azure Container Apps with a system-assigned managed identity that calls Microsoft Foundry (no API keys).
# Prerequisites: az CLI logged in (`az login`), a Foundry project with a deployed chat model.
#
#   export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
#   export FOUNDRY_MODEL="<deployment name>"
#   export FOUNDRY_RESOURCE_ID="/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<resource>"
#   ./infra/deploy.sh
set -euo pipefail
: "${FOUNDRY_PROJECT_ENDPOINT:?set FOUNDRY_PROJECT_ENDPOINT}" "${FOUNDRY_MODEL:?set FOUNDRY_MODEL}" "${FOUNDRY_RESOURCE_ID:?set FOUNDRY_RESOURCE_ID}"
RG="${RG:-rg-insight-engine}"; LOC="${LOC:-eastus}"; ENV_NAME="${ENV_NAME:-insight-env}"; APP="${APP:-insight-engine}"

az extension add --name containerapp --upgrade -y >/dev/null
az group create -n "$RG" -l "$LOC" >/dev/null

# Builds the image in Azure (ACR build), creates the environment and the app, exposes port 8000 publicly (judges need a link).
az containerapp up -n "$APP" -g "$RG" -l "$LOC" --environment "$ENV_NAME" --source . --ingress external --target-port 8000

az containerapp identity assign -n "$APP" -g "$RG" --system-assigned >/dev/null
PRINCIPAL=$(az containerapp show -n "$APP" -g "$RG" --query identity.principalId -o tsv)

# Let the app's identity call models in the Foundry resource.
az role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" --scope "$FOUNDRY_RESOURCE_ID" >/dev/null || true
az role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role "Azure AI User" --scope "$FOUNDRY_RESOURCE_ID" >/dev/null || true

az containerapp update -n "$APP" -g "$RG" \
  --set-env-vars FOUNDRY_PROJECT_ENDPOINT="$FOUNDRY_PROJECT_ENDPOINT" FOUNDRY_MODEL="$FOUNDRY_MODEL" \
  --min-replicas 1 >/dev/null     # keep one replica warm: the background cache warm-up runs once at start

URL=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "Deployed: https://$URL"
echo "Check:    curl https://$URL/health   (llm.enabled should be true; role assignments can take a few minutes)"
echo "          curl https://$URL/agents"
