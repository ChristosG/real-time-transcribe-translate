docker  run --rm -it --net host --ulimit memlock=-1 --ulimit stack=67108864 \
            		--security-opt=label=disable --security-opt seccomp=unconfined \
            		--tmpfs /tmp:exec --user root \
                --runtime=nvidia \
                --ipc=host \
              	-p8000:8000 -p8001:8001 -p8002:8002 \
                --name lw_latest \
                -v /home/chris/engines:/engines \
                -v /home/chris/tensorrtllm_backend:/trtback \
                llama_whisper /bin/bash
