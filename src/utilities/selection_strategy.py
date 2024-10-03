from abc import ABC, abstractmethod
from typing import Dict, List
import os
import sys 

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

from mab.mpts import MPTS


class ProtocolSelectionStrategy(ABC):
    @abstractmethod
    def select_protocols(self, all_protocols: Dict[int, int]) -> Dict[int, int]:
        pass

class MPTSStrategy(ProtocolSelectionStrategy):
    def __init__(self, all_protocols, mpts_config: Dict, environment, cm, k: int):
        self.mpts = MPTS(
            arms=all_protocols,
            k=k,
            T=int(mpts_config['T']),
            thread=environment.kernel_thread,
            net_channel=cm.netlink_communicator,
            step_wait=mpts_config['step_wait']
        )

    def select_protocols(self) -> Dict[int, int]:
        pool = self.mpts.run()
        return {action: int(p_id) for action, p_id in enumerate(pool)}

class ManualSelectionStrategy(ProtocolSelectionStrategy):
    def __init__(self, selected_protocols: List[int]):
        self.selected_protocols = selected_protocols

    def select_protocols(self) -> Dict[int, int]:
        return {i: proto_id for i, proto_id in enumerate(self.selected_protocols)}
