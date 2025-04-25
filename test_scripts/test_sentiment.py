from transformers import pipeline
analyzer = pipeline("sentiment-analysis")
print(analyzer("Apple stock hits all-time high"))


###NEED TO INSTALL PYTORCH OR TENSORFLOW
# !pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118