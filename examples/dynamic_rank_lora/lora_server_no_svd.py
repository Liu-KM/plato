"""
A federated learning server using LoRA fine-tuning.

To run this example:

python examples/lora/lora_server.py -c examples/lora/server.yml
"""

import logging

from plato.servers import fedavg_lora, fedavg

from lora_utils import LoraModel, DataSource, Trainer, Algorithm
# from plato.trainers.basic import Trainer


# class Server(fedavg_lora.Server):
class Server(fedavg.Server):
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
        # pylint: disable=unused-argument

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)
        

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server

        return avg_update



def main():
    server = Server(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server.run()


if __name__ == "__main__":
    main()
