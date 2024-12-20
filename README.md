# Lux-AI-S3

## Getting started

1. Create a New Environment

```
conda create -n lux-s3 python=3.11
```

2. Activate the Environment
```
conda activate lux-s3
```

3. Clone the Repository
```
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3/
```

4. Install the Package
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
