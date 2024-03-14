""" 
A federated learning client using LoRA fine-tuning.


To run this example:

python examples/lora/lora_client.py -c examples/lora/client.yml -i <client_id>
"""

import asyncio
import logging

from plato.clients import simple
from plato.config import Config
from lora_utils import LoraModel, DataSource,Trainer, Algorithm
# from plato.trainers.basic import Trainer
from peft import (
    get_peft_model,
    LoraConfig,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)



class Client(simple.Client):
    """A client using LoRA fine-tuning."""

    def __init__(self, model=None, datasource=None, trainer=None, algorithm=None):
        super().__init__(
            model=model, datasource=datasource, trainer=trainer, algorithm=algorithm
        )
        logging.info("A LoRA client has been initialized.")
    # def _load_payload(self, server_payload) -> None:
    #     weights = server_payload[0]
    #     rank = server_payload[1]
    #     lora_config = Config().parameters.lora
    #     rank = min(rank,lora_config.r)
    #     temp_model = self.trainer.model.base_model.unload()
    #     self.trainer.model.base_model = get_peft_model(temp_model, LoraConfig(**lora_config._asdict()))
    #     for key in weights.keys():
    #         if "lora_A" in key:
    #             weights[key] = weights[key][:rank,:]
    #         elif "lora_B" in key:
    #             weights[key] = weights[key][:,:rank]
    #     self.algorithm.load_weights(weights)
    def _load_payload(self, server_payload) -> None:
        weights = server_payload[0]
        rank = server_payload[1]
        self.algorithm.reset_rank(rank)
        for key in weights.keys():
            if "lora_A" in key:
                weights[key] = weights[key][:rank,:]
            elif "lora_B" in key:
                weights[key] = weights[key][:,:rank]
        self.algorithm.load_weights(weights)

    

def main():
    client = Client(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    client.configure()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())


if __name__ == "__main__":
    main()
