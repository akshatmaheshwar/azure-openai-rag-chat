metadata description = 'Creates an Azure AI Search service.'
param name string
param location string = resourceGroup().location
param tags object = {}

@allowed([ 'free', 'basic' ])
param sku string = 'free'

@description('Disable API keys so that only Microsoft Entra ID (RBAC) authentication is accepted.')
param disableLocalAuth bool = true

@allowed([ 'enabled', 'disabled' ])
param publicNetworkAccess string = 'enabled'

resource search 'Microsoft.Search/searchServices@2025-05-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    hostingMode: 'Default'
    publicNetworkAccess: publicNetworkAccess
    // authOptions can only be set when local auth is enabled
    disableLocalAuth: disableLocalAuth
    authOptions: disableLocalAuth ? null : {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
  }
}

output endpoint string = 'https://${search.name}.search.windows.net'
output id string = search.id
output name string = search.name
