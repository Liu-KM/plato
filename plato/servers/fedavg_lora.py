
"""
A federated learning server using federated averaging to aggregate updates after homomorphic encryption.
"""
import asyncio
import logging
import os

from plato.config import Config
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor, fonts
from functools import reduce
from plato.servers import fedavg
from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)
import torch


class Server(fedavg.Server):

    def convert_to_full_rank(self, weights):
        layer_list = []
        for name in weights.keys():
            if "lora_A" in name or "lora_B" in name:
                layer = name.rsplit('.',2)[0]
                layer_list.append(layer)
        layer_list = list(set(layer_list))

        for name in layer_list:
            lora_A_weight = weights.pop(name+".lora_A.weight")
            lora_B_weight = weights.pop(name+".lora_B.weight")
            delta_W = lora_A_weight.t()@lora_B_weight.t()
            weights[(name+".AB")] = delta_W

            # U,S,V = torch.svd()
            # U_truncated = U[:,:r]
            # S_truncated = U[:r]
            # V_truncated = U[:,:r]
            # weights[(name+".lora_A.weight")] = U_truncated@torch.diag(S_truncated).t()
            # weights[(name+".lora_B.weight")] = V_truncated
        return weights

    def convert_to_low_rank(self,weights):
        r = Config().parameters.lora.r
        for name in list(weights.keys()):
            if ".AB" in name:
                U,S,V = torch.svd(weights.pop(name))
                name = name.rsplit('.',1)[0]
                U_truncated = U[:,:r]
                S_truncated = S[:r]
                V_truncated = V[:,:r]
                weights[(name+".lora_A.weight")] = (U_truncated@torch.diag(S_truncated)).t()
                weights[(name+".lora_B.weight")] = V_truncated
        return weights
            

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)
        full_rank_weights = [ self.convert_to_full_rank(weight)  for weight in weights_received]

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in full_rank_weights[0].items()
        }

        for i, update in enumerate(full_rank_weights):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server

        return self.convert_to_low_rank(avg_update)

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        weights_received = [update.payload for update in self.updates]

        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        # Extract the current model weights as the baseline
        baseline_weights = self.algorithm.extract_weights()

        if hasattr(self, "aggregate_weights"):
            # Runs a server aggregation algorithm using weights rather than deltas
            logging.info(
                "[Server #%d] Aggregating model weights directly rather than weight deltas.",
                os.getpid(),
            )
            updated_weights = await self.aggregate_weights(
                self.updates, baseline_weights, weights_received
            )

            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)
        else:
            # Computes the weight deltas by comparing the weights received with
            # the current global model weights
            baseline_weights = self.convert_to_full_rank(baseline_weights)
            weights_received = self.convert_to_full_rank(weights_received)


            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )
            # Runs a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[Server #%d] Aggregating model weight deltas.", os.getpid())
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)
            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)

        # The model weights have already been aggregated, now calls the
        # corresponding hook and callback
        self.weights_aggregated(self.updates)
        self.callback_handler.call_event("on_weights_aggregated", self, self.updates)

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy, self.accuracy_std = self.get_accuracy_mean_std(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server
            logging.info("[%s] Started model testing.", self)
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model perplexity: {self.accuracy:.2f}\n"
                )
            )
        else:
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n"
                )
            )

        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)

