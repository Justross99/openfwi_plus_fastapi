# app/core/api_metadata.py
"""
API metadata for the FastAPI application, including description and tags.
"""

api_description = """
The OpenFWI Model Server API provides access to trained Unified Variational Autoencoder (VAE) models
for seismic velocity inversion.

You can use this API to:
- ✅ **Check API health**
- 📝 **List available models**
- ℹ️ **Get detailed configuration and architecture summaries** for each model
- 🚀 **Perform inference** to generate velocity maps from seismic data

Models are loaded on-demand and cached for efficiency.
"""

tags_metadata = [
    {
        "name": "Inference",
        "description": "Endpoints for performing model inference.",
    },
    {
        "name": "Model Information",
        "description": "Endpoints for querying model configurations, architectures, and availability.",
    },
    {
        "name": "Health & Status",
        "description": "Endpoints for monitoring the API's health.",
    },
]
