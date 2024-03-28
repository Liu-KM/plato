"""
A federated learning server using LoRA fine-tuning.

To run this example:

python examples/lora/lora_server.py -c examples/lora/server.yml
"""

import logging
import torch
from plato.config import Config
from plato.servers import fedavg_lora, fedavg

from lora_utils import LoraModel, DataSource, Trainer, Algorithm
# from plato.trainers.basic import Trainer


class Server(fedavg_lora.Server):
# class Server(fedavg.Server):
    """A federated learning server using LoRA fine-tuning."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        logging.info("A LoRA server has been initialized.")

    def save_to_checkpoint(self):
        logging.info("Skipping checkpoint.")
    
    def convert_to_low_rank(self,weights):
        r = Config().parameters.lora.r
        for name in list(weights.keys()):
            if ".AB" in name:
                U,S,V = torch.svd(weights.pop(name))
                name = name.rsplit('.',1)[0]
                U_truncated = U[:,:r]
                S_truncated = S[:r]
                V_truncated = V[:,:r]
                # A = U@S  ; B = V
                # weights[(name+".lora_A.weight")] = (U_truncated@torch.diag(S_truncated)).t()
                # weights[(name+".lora_B.weight")] = V_truncated
                # A = U ; B = V@S
                weights[(name+".lora_A.weight")] = (U_truncated).t()
                weights[(name+".lora_B.weight")] = V_truncated@torch.diag(S_truncated).t()
                
        return weights

def main():
    server = Server(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server.run()


if __name__ == "__main__":
    main()
