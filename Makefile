PROJECT_NAME=hand-sign
VERSION=0.0.1

IMAGE_NAME=$(PROJECT_NAME):$(VERSION)
IMAGE_NAME_P310=$(PROJECT_NAME)_p310:$(VERSION)
IMAGE_NAME_P310_CPU=$(PROJECT_NAME)_p310_cpu:$(VERSION)
CONTAINER_NAME=--name=$(PROJECT_NAME)

GPUS=--gpus=all  # Specifies which GPUs the container can see
NET=--net=host
IPC=--ipc=host
BUILD_NET=--network=host

.PHONY: all build stop run logs

all: build stop run logs

build:
	docker build $(BUILD_NET) -t $(IMAGE_NAME) -f Dockerfile .

build-p310:
	docker build $(BUILD_NET) -t $(IMAGE_NAME_P310) -f Dockerfile-p310 .

build-p310-cpu:
	docker build $(BUILD_NET) -t $(IMAGE_NAME_P310_CPU) -f Dockerfile-p310-cpu .

stop:
	docker stop $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)

kill:
	docker kill $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)
	docker rm $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)

run:
	docker run --rm -it $(GPUS) $(NET) $(IPC) \
		-v $(shell pwd):/workdir/ \
		$(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		bash

run-p310:
	docker run --rm -it $(GPUS) $(NET) $(IPC) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(HOME)/.Xauthority:/root/.Xauthority:rw \
		-e DISPLAY=$(shell echo ${DISPLAY}) \
		-v $(shell pwd):/workdir/ \
		$(CONTAINER_NAME) \
		$(IMAGE_NAME_P310) \
		bash

run-p310-cpu:
	docker run --rm -it $(GPUS) $(NET) $(IPC) \
		-v $(shell pwd):/workdir/ \
		$(CONTAINER_NAME)_cpu \
		$(IMAGE_NAME_P310_CPU) \
		bash

run-x11:
	docker run --rm -it $(GPUS) $(NET) $(IPC) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(HOME)/.Xauthority:/root/.Xauthority:rw \
		-e DISPLAY=$(shell echo ${DISPLAY}) \
		-v $(shell pwd):/workdir/ \
		$(CONTAINER_NAME)_x11 \
		$(IMAGE_NAME) \
		bash

logs:
	docker logs -f $(PROJECT_NAME)
