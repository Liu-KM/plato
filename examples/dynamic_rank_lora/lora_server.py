"""
A federated learning server using LoRA fine-tuning.

To run this example:

python examples/lora/lora_server.py -c examples/lora/server.yml
"""

import logging
import random
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
    def customize_server_payload(self, payload):
        if not hasattr(self, "client_data_size"):
            self.client_data_size = {}
        for update in self.updates:
            self.client_data_size[update.client_id] = update.report.num_samples
        rank = 0
        sample_size=-1
        if self.selected_client_id in self.client_data_size:
            sample_size = self.client_data_size[self.selected_client_id]
        #get the rank of the client size in the list of clients
        for size in self.client_data_size.values():
            if size<sample_size:
                rank+=1
        
        if len(self.updates) == 0 or sample_size==-1:
            r = Config().parameters.lora.r
        else:
            r = rank/len(self.updates)*Config().parameters.lora.r
            r= int(r)
        
        logging.info(f"Client:{self.selected_client_id} Datasize:{sample_size} Rank:{r}")
        return [payload,r]
    def save_to_checkpoint(self):
        logging.info("Skipping checkpoint.")


def main():
    server = Server(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server.run()


if __name__ == "__main__":
    main()
