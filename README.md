# RecruitBot

## Requirements

Recruit Bot requires Python 3.5+ and the following python libs:

```bash
pip install webcollector
pip install flask
pip install BeautifulSoup4

# CPU
pip install tensorflow

# GPU
pip install tensorflow-gpu
```

## Run

The __recruit_bot.py__ is used for both training and serving.

### Training

For training, you should set `training = True` in recruit_bot.py and run __recruit_bot.py__.


### Serving

For serving, you should set `training = False` in recruit_bot.py and run __recruit_bot.py__.

After that, you can visit [http://127.0.0.1:5002](http://127.0.0.1:5002) to generate posts.



