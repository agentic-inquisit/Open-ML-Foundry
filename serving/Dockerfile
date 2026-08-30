# Use official TorchServe image
FROM pytorch/torchserve:latest-cpu

# Switch to root for installing dependencies if needed
USER root
RUN pip install torchvision pillow

# Copy handler and model artifacts
# In a real pipeline, we would copy the .mar file here
COPY handler.py /home/model-server/handler.py
# COPY model.pt /home/model-server/model.pt

# Set working directory
WORKDIR /home/model-server

# Expose ports for management and inference
EXPOSE 8080 8081 8082

# TorchServe command to start the server
ENTRYPOINT ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models", "resnet50=resnet50.mar", "--ts-config", "config.properties"]
