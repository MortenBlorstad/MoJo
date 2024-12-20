# Lux-AI-S3

## Getting started

1. Install Mamba (if not already installed)
```
conda install mamba -n base -c conda-forge
```

2. Create a New Environment

```
mamba create -n "lux-s3" "python==3.11"
```

3. Activate the Environment
```
mamba activate lux-s3
```

4. Clone the Repository
```
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3/
```

5. Install the Package
```
cd Lux-Design-S3/src
pip install -e .
```

To verify your installation, you can run a match between two random agents:
```
luxai-s3 --help
```
```
luxai-s3 path/to/bot/main.py path/to/bot/main.py --output replay.json
```
Then upload the replay.json to the online visualizer here: https://s3vis.lux-ai.org/ (a link on the lux-ai.org website will be up soon)
