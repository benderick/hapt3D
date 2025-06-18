.PHONY: test

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
CONFIG ?= 'config/config_full.yaml'
CHECKPOINT ?= ''
RUN_IN_CONTAINER = docker compose run -e CUDA_VISIBLE_DEVICES=$(GPUS) hapt3d

build:
	COMPOSE_DOCKER_CLI_BUILD=1 docker compose build hapt3d --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)
train:
	$(RUN_IN_CONTAINER) python3 train.py --config $(CONFIG)
test:
	$(RUN_IN_CONTAINER) python3 test.py --weights $(CHECKPOINT)

shell:
	$(RUN_IN_CONTAINER) bash
freeze_requirements:
	pip-compile requirements.in > requirements.txt