export no_proxy="localhost,127.0.0.1"

CUDA_VISIBLE_DEVICES=0 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 0  &
CUDA_VISIBLE_DEVICES=0 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 1  &
CUDA_VISIBLE_DEVICES=0 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 2  &
CUDA_VISIBLE_DEVICES=0 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 3  &
CUDA_VISIBLE_DEVICES=0 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 4  &
CUDA_VISIBLE_DEVICES=1 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 5  &
CUDA_VISIBLE_DEVICES=1 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 6  &
CUDA_VISIBLE_DEVICES=1 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 7  &
CUDA_VISIBLE_DEVICES=1 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 8  &
CUDA_VISIBLE_DEVICES=1 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 9  &
CUDA_VISIBLE_DEVICES=2 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 10  &
CUDA_VISIBLE_DEVICES=2 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 11  &
CUDA_VISIBLE_DEVICES=2 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 12  &
CUDA_VISIBLE_DEVICES=2 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 13  &
CUDA_VISIBLE_DEVICES=2 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 14  &
CUDA_VISIBLE_DEVICES=3 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 15  &
CUDA_VISIBLE_DEVICES=3 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 16  &
CUDA_VISIBLE_DEVICES=3 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 17  &
CUDA_VISIBLE_DEVICES=3 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 18  &
CUDA_VISIBLE_DEVICES=3 python /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/lora_client.py -c /home/l84346065/plato_env/plato/examples/svd_lora_SEQ_CLS/client.yml -i 19  &

wait