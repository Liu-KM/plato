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
        rank = 0
        for update in self.updates:
            if update.client_id == self.selected_client_id:
                sample_size = update.report.num_samples
                break
        for update in self.updates:
            if update.report.num_samples >= sample_size:
                rank += 1
        r = rank/len(self.updates)*Config().parameters.lora.r
        return [payload,rank]
    def save_to_checkpoint(self):
        logging.info("Skipping checkpoint.")


def main():
    server = Server(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server.run()


if __name__ == "__main__":
    main()
