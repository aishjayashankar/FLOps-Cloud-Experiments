import flops_infra_drift.consts as consts
import pickle

from concurrent.futures import ThreadPoolExecutor
from flops_infra_drift.dropped_client_replacer import get_dropped_client_parameters
from flwr.common import FitRes, FitIns, Parameters
from flwr.server.client_proxy import ClientProxy
from typing import Union
from typing import Optional, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import  FedAvg
from flwr.server.strategy.strategy import Strategy

# pylint: disable=line-too-long
class DIWS(Strategy):
    """Distribution Informed Weight Substitution (DIWS) strategy.

    Wrapper for any Flower Strategy that substitutes weights for dropped clients
    Parameters
    ----------
    aggregator_strategy : Strategy
        The strategy to be wrapped. This strategy will be used for the actual
        training and evaluation of the clients. The DIWS strategy will only
        substitute weights for dropped clients during the aggregation phase.
    global_parameters : Internally managed
        To be re-used during subset training for weight substitution
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        aggregator_strategy: Strategy = FedAvg
    ) -> None:
        super().__init__()
        self.aggregator_strategy = aggregator_strategy
        self.global_parameters = None
        self.label_distribution = {}

    def __repr__(self) -> str:
        return repr(self.aggregator_strategy)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        self.global_parameters = self.aggregator_strategy.initialize_parameters(client_manager)
        return self.global_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        return self.aggregator_strategy.evaluate(server_round, parameters)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        self.global_parameters = parameters
        return self.aggregator_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return self.aggregator_strategy.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results from clients, substituting dropped clients if necessary."""

        # Initialize label distribution for the first round
        if server_round == 1:
            for _, fitres in results:
                client_label_distribution = pickle.loads(fitres.metrics.get("label_distribution"))
                self.label_distribution[client_label_distribution[0]] = client_label_distribution[1]
        print(f"Label distribution for clients: {self.label_distribution}")

        self.substitute_dropped_clients(server_round, results, failures)

        return self.aggregator_strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        return self.aggregator_strategy.aggregate_evaluate(server_round, results, failures)

    def substitute_dropped_clients(self,
                                server_round: int,
                                results: list[tuple[ClientProxy, FitRes]],
                                failures: list[Union[tuple[ClientProxy, FitRes], BaseException]]) -> None:
        """Substitute dropped clients with subset training, if required"""
        fitIns = FitIns(
            parameters=self.global_parameters,
            config={"custom_rpc": "handle_missing_clients"}
        )
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(result[0].fit,
                            fitIns,
                            consts.SUBSTITUTION_TIMEOUT,
                            server_round)
                        for result in results]
            outputs = [f.result() for f in futures]

        # print(f"Results' length before substitution: {len(results)}")
        # if (consts.CLIENT_DROP_ROUND_START <= server_round < consts.CLIENT_DROP_ROUND_END):
        #     get_dropped_client_parameters(results)
        # print(f"Results' length after substitution: {len(results)}")