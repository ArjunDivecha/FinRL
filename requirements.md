conda create -n finrl python=3.9 -y
conda activate finrl

# Core dependencies
pip install finrl[full]
pip install stable-baselines3[extra]
pip install yfinance matplotlib pandas
pip install torch torchvision torchaudio  # includes MPS acceleration for Apple Silicon