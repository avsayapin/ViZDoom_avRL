# Deep Reinforcement Learning in ViZDoom

Repository for "Deep Reinforcement Learning in VizDoom" thesis.
Repository contains scenarios and models to test audiovisual RL models.
## Models

APPO

IMPALA

## Requirements

Python 3.8.5

ViZDoom 1.1.13

Pytorch 1.6

Torchaudio

## Installation guide

Install miniconda with Python 3.8.5

Create new conda environment <somename>

```
#Don't forget to change <somename> to something else
conda env create --name <somename>
conda activate <somename>
```

Install ViZDoom build requirements

```
conda install -c conda-forge boost cmake gtk2 sdl2 fluidsynth openal-soft
#codna remove fontconfig # if it conflicts with system libraries
git clone https://github.com/mwydmuch/ViZDoom.git --recurse-submodules
cd ViZDoom
python setup.py build && python setup.py install
cd ..
```
Download sample-factory

```
git clone https://github.com/alex-petrenko/sample-factory.git
```
Change environment name in environment.yml to <somename>

Update your environment
```
conda remove fluidsynth # conflicts with opencv
cd sample-factory
conda env update -f environment.yml
```
Replace folder sample_factory/envs/doom with folder doom from this repository

Add train.py to sample_factory folder

## Experiments
Use this command to start APPO training in Music Recognition scenario
```
python -m sample_factory.train --env=doomsound_music_recognition --experiment=appo --encoder_custom=vizdoomSoundFFT --train_for_env_steps=1500000000 --seed=0 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0
```
## References

Sample factory: https://github.com/alex-petrenko/sample-factory

Agents That Listen: https://github.com/hegde95/Agents_that_Listen