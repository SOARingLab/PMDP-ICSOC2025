# ----------------------------------------------------------------------------------
# Dockerfile for P-MDP Framework
# ----------------------------------------------------------------------------------

# Step 1: Use an official NVIDIA CUDA image as the base
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Step 2: Prepare the environment
ENV DEBIAN_FRONTEND=noninteractive
ENV IS_IN_DOCKER=true
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
# Initialize the Conda shell environment to ensure subsequent conda commands work correctly
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Step 3: Set the working directory
WORKDIR /app

# Step 4: Accept Anaconda's Terms of Service (ToS)
RUN conda tos accept

# Step 5: Copy the environment file and create the Conda environment
# Note: This step can be time-consuming as Conda resolves and downloads packages.
COPY pmdp_conda.yaml .
RUN conda env create -f pmdp_conda.yaml && conda clean -afy

# Step 6: Copy all remaining project files
COPY . .

# Step 7: Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh



# Step 8: Pre-install all local .vsix extensions.
COPY vscode_extensions/ /tmp/vscode_extensions/
RUN for ext in /tmp/vscode_extensions/*.vsix; do \
        echo "--- Installing extension: $ext ---"; \
        code-server --install-extension "$ext" || true; \
    done && \
    rm -rf /tmp/vscode_extensions



# Step 9: Expose the default port for code-server
EXPOSE 8080

# Step 10: Set the container's startup command
# This ensures that the default shell will automatically activate the correct Conda environment.
CMD ["conda", "run", "-n", "Py311Env4PMDP", "bash", "-c", "code-server --bind-addr 0.0.0.0:8080 --auth none ."]