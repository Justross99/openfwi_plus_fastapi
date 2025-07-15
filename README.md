# OpenFWI: A Production-Grade Framework for Deep Learning in Full Waveform Inversion

## Project Motivation & Philosophy

This project was built as a **reference architecture** to demonstrate the principles and best practices required to build and deploy a modern, end-to-end deep learning system.

**The Problem:** Many data science projects exist in a research vacuum, often confined to Jupyter notebooks, making them difficult to productionize. Conversely, many backend engineering projects lack the complexity of handling computationally intensive ML workloads.

**The Solution:** This project was designed to bridge this gap. It is a holistic framework that seamlessly integrates two distinct but connected worlds:
1.  **The ML Research System:** A flexible, high-performance environment for iterative model development.
2.  **The Production Inference Service:** A robust, scalable, and secure API for delivering the model's value to users.

The goal is to provide a blueprint for MLOps best practices, showcasing the architectural thinking required to deliver a complete, reliable, and maintainable deep learning solution.

## Architectural Deep Dive: Key Design Decisions

This section explains the "why" behind the project's core architectural decisions.

### 1. The Dual-Component Architecture: Bridging Research & Production

The project is intentionally split into two main parts (`train/`, `model/` vs. `app/`) to mirror the real-world workflow of a modern MLOps team. This structure allows for a clean separation of concerns between model experimentation and production serving, enabling specialized development while ensuring seamless integration.

### 2. The Machine Learning System: Engineered for Excellence

#### Why JAX/Flax?
- **Performance & Modern Hardware:** We chose JAX/Flax over other frameworks to leverage its state-of-the-art performance on modern accelerators (GPUs/TPUs). Its functional, composable transformations (`jit`, `vmap`, `grad`) provide fine-grained control for highly optimized code.
- **Code Quality & Predictability:** JAX's functional paradigm (pure functions) makes the ML code more predictable, easier to debug, and less prone to side effects—a mature software engineering principle applied to ML.

#### Memory-Efficient Data Handling
- **The Challenge:** Seismic datasets are often too large to fit in memory.
- **Our Solution:** We built a custom dataloader using **Hugging Face `datasets`** with a Python generator backend. This creates a memory-mapped dataset that streams data from disk on-the-fly, allowing us to process massive datasets without exhausting system RAM. This demonstrates the ability to engineer solutions for real-world data scale constraints.

#### Modular by Design: Fostering Rapid Experimentation
- **The Philosophy:** Models and loss functions are not monolithic scripts. They are built from modular, composable components in the `model/` and `criterion/` directories.
- **The Benefit:** This design, an application of the **Separation of Concerns** principle, makes the system highly extensible. A researcher can add a novel decoder or loss function without rewriting the entire training pipeline, accelerating experimentation and improving long-term maintainability.

### 3. The Production Inference API: Built for Reliability and Scale

#### Code Structure: Separation of Concerns in Practice
- **The Decision:** The `app/` directory is strictly organized into `routers`, `services`, and `core`.
- **The Justification:**
    - `routers/`: Handles the "web layer" (HTTP requests/responses).
    - `services/`: Encapsulates the core "business logic" (e.g., loading a model).
    - `core/`: Contains cross-cutting concerns (configuration, security).
- **The Benefit:** This decoupling makes the system highly **maintainable and testable**. We can test API endpoints by mocking the service layer, and we can change the model loading strategy (e.g., from disk to S3) by modifying only the service, without touching the API code.

#### Dependency Injection: The Key to Testability
- **The Decision:** We heavily leverage FastAPI's `Depends` system.
- **The Justification:** This powerful design pattern "inverts control," providing dependencies to functions instead of having them create their own.
- **The Benefit:** This is the secret to our robust testing strategy. In `tests/api/`, we can easily **override dependencies** to inject mock models, allowing us to test API logic in complete isolation.

#### A Pragmatic Caching Strategy
- **The Decision:** We cache model information endpoints (`/models`) but **do not** cache the `/predict` endpoint. We use an in-memory cache for local development and a **Redis** backend for the production Docker Compose setup.
- **The Justification:**
    - **Why cache model info?** This data is static and requested frequently. Caching it avoids redundant computation and improves responsiveness.
    - **Why not cache predictions?** The input data for each prediction is expected to be unique. Caching would provide no benefit and would waste memory storing large, one-off requests and responses.
    - **Why Redis for Production?** While an in-memory cache is sufficient for single-instance development, a production environment runs multiple, stateless API instances. A centralized cache like Redis is essential to ensure **cache consistency** across all instances and to persist the cache even if a container restarts. This demonstrates architectural foresight for building scalable systems.

#### Security as a First-Class Citizen
- **CORS (Cross-Origin Resource Sharing):** We included `CORSMiddleware` with the foresight that the API will be consumed by a web-based frontend on a different domain.
- **API Key & Rate Limiting:** These are essential protections for a computationally expensive service. The API key prevents unauthorized use, while rate limiting protects against abuse and ensures service availability.

#### Observability by Design
- **Structured JSON Logging:** We chose JSON for logs because they are destined for machines, not humans. Structured logs can be ingested, parsed, and indexed by platforms like **Datadog or Splunk**, enabling powerful automated searching, filtering, and alerting.
- **Prometheus Metrics:** We expose a `/metrics` endpoint because Prometheus is the industry standard for monitoring. This provides real-time data on application health (latency, traffic, errors) for visualization on Grafana dashboards and automated alerting.

## Comprehensive Testing Strategy

A robust testing suite is non-negotiable for a production-grade system. Our philosophy is to test what matters at the right level, ensuring both the individual components and the integrated system work as expected. The tests are split into two main categories:

- **`tests/api/`**: Integration tests for the FastAPI application.
- **`tests/ml/`**: Unit tests for the machine learning components.

### API Testing (`tests/api`)
We use `pytest` and `httpx.AsyncClient` to test the API layer. The focus is on verifying the API contract: correct status codes, response schemas, and error handling.

Crucially, we leverage **FastAPI's dependency injection** to mock the ML model and service layers. This allows us to test the API logic (routing, validation, security) in complete isolation, resulting in fast, reliable, and deterministic tests that don't require loading a real model.

### ML Component Testing (`tests/ml`)
These are unit tests that validate the correctness of individual pieces of our machine learning system. This includes testing:
- **Data Scalers:** Ensuring that data transformations are calculated and applied correctly.
- **Dataloaders:** Verifying that data is yielded in the correct shape and format.
- **Model Components:** Confirming that the forward pass of individual Flax modules produces outputs of the expected shape.

### Running Tests
To run the full test suite, simply execute:
```bash
pytest
```

## Project Structure
```
.
├── app/                  # Production FastAPI Inference Server
│   ├── core/             # Core logic: config, logging, security
│   ├── routers/          # API endpoints (inference, models)
│   └── services/         # Services for model loading and registry
├── criterion/            # Composable loss functions for training
├── data/                 # Memory-efficient dataloaders and data scalers
├── model/                # Modular Flax model components (backbones, decoders)
├── output/               # Default directory for training runs, models, and logs
├── tests/                # Unit and integration tests
├── train/                # Training scripts and configuration
├── .env.example          # Environment variable template
├── docker-compose.yml    # Docker Compose for multi-container setups
├── dockerfile            # Multi-stage Dockerfile for the inference API
├── pyproject.toml        # Project dependencies and metadata
└── README.md             # You are here
```

## The Machine Learning System

Our ML system is designed for flexibility and performance, enabling rapid iteration on complex deep learning models.

### Efficient Data Handling (`data/`)
Seismic datasets can be enormous. To handle this, our dataloader in `data/dataloader.py` uses **Hugging Face `datasets`** backed by a Python generator. This approach creates a memory-mapped dataset that reads data chunks directly from NumPy files on disk as needed, preventing memory exhaustion and enabling efficient processing of terabyte-scale data.

### Modular Model Architecture (`model/`)
We believe in building models from reusable, interchangeable parts. The `model/` directory is structured to reflect this philosophy:
- `backbone/`: Contains the core feature extractors (e.g., U-Net encoders).
- `decoder/`: Holds components that reconstruct the velocity map from a latent representation.
- `latents/`: Defines the latent space, such as a Variational Autoencoder (VAE) sampling head.
- `full_model_defs/`: Assembles these components into complete Flax `nn.Module` definitions.

This design allows you to create novel architectures simply by combining different components.

### Flexible Loss Functions (`criterion/`)
FWI often requires optimizing for multiple objectives simultaneously. Our `CombinedLoss` wrapper in `criterion/combined_loss.py` makes this easy. Individual loss functions (e.g., `MSE`, `SSIM`, `KLDivergence`) are defined as separate modules. The `CombinedLoss` class takes a list of these criteria and their corresponding weights from the training configuration to create a single, weighted loss function.

## Getting Started: Training and Deployment

This section provides a complete walkthrough for getting the project running, from acquiring the data and training a model to deploying the inference API.

### 1. Acquiring the OpenFWI Dataset

This project is designed to work with the **OpenFWI** datasets, which are a collection of large-scale, open-source datasets for Full-Waveform Inversion research.

- **Download the Data:** The datasets are publicly available from the SMILE team. We recommend starting with the `OpenFWI-A` dataset. You can find and download it here: [https://smileunc.github.io/projects/openfwi/datasets](https://smileunc.github.io/projects/openfwi/datasets)
- **Data Structure:** After downloading and extracting, you should have a directory containing NumPy (`.npy`) files for seismic data and velocity models.

### 2. Configuring the Environment

1.  **Clone the repository, create a virtual environment, and install dependencies:**
    ```bash
    git clone <repository-url> && cd Openfwi_src
    python -m venv .venv && source .venv/bin/activate
    uv pip install -e .
    ```
2.  **Set up environment variables:**
    Copy `.env.example` to `.env` and customize the values.
    ```bash
    cp .env.example .env
    ```
    **Crucially, update `MODEL_ARTIFACTS_DIR` in your `.env` file to point to the directory where you downloaded the OpenFWI dataset.** This path will also be used to save training outputs.

### 3. Training a New Model

The entire training process is controlled by a central configuration file and a main training script.

#### Configure Your Training Run
Open `train/config.py`. This file, using `ml_collections.ConfigDict`, is the single source of truth for your experiment. Here you can:
- Set the learning rate, batch size, and number of training epochs.
- Select the model architecture by choosing from the components in the `model/` directory.
- Define the loss function by combining different criteria from `criterion/` and setting their weights.

#### Launch the Training Loop
Once your configuration is set, start the training process with:
```bash
uv run train.train_loop
```
The script will create a timestamped output directory within the path specified by `MODEL_ARTIFACTS_DIR`. Inside, you will find:
- **TensorBoard logs** for monitoring training metrics in real-time.
- **Model checkpoints** saved periodically.

### 4. Running the Inference API

Once you have a trained model, you can serve it using the production-ready API.

#### Locally for Development
To run the API server with hot-reloading:
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`, with interactive documentation at `http://127.0.0.1:8000/docs`.

#### Using Docker Compose (Recommended)
For a production-like environment that includes the API service and a Redis cache, use Docker Compose.

1.  **Build the services:**
    ```bash
    docker-compose build
    ```

2.  **Run the application stack:**
    ```bash
    docker-compose up -d
    ```
    This command starts both the `api` and `redis` containers in the background. The API will be available at `http://127.0.0.1:8000`.

## Future Work & Roadmap

This project provides a solid foundation, but the journey of a production system is never truly finished. Here are some of the planned enhancements to further improve the robustness and capabilities of this framework:

-   **CI/CD Automation:** Implement a full CI/CD pipeline using GitHub Actions to automate testing, building Docker images, and deploying to a cloud environment.
-   **Infrastructure as Code (IaC):** Use Terraform or a similar tool to codify the deployment infrastructure (e.g., cloud instances, networking, managed Redis), enabling reproducible and version-controlled environments.
-   **Enhanced Experiment Tracking:** Integrate a dedicated experiment tracking tool like MLflow or Weights & Biases to systematically log parameters, metrics, and artifacts for every training run, improving reproducibility and insight.
-   **Deeper Monitoring & Alerting:** Expand the current observability stack with more granular application performance monitoring (APM) and set up automated alerts for critical events like high error rates, latency spikes, or resource saturation.
-   **Model Zoo Expansion:** Add more pre-trained models and architectures to the `model/` directory, turning the project into a more comprehensive library for FWI research.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
