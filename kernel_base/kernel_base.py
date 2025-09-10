from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Tuple

import torch

from .registry import register_kernel


@register_kernel
class KernelBase(ABC):

    _default_params: Dict[str, Any] = {}
    _kernel_name: Optional[str] = None
    _key: Optional[str] = None

    def __init__(self, device: torch.device):
        self.device = device
        self.params = self.__class__._default_params.copy()
        self.inputs = None

    @classmethod
    def get_param_list(cls) -> List[str]:
        return list(cls._default_params.keys())

    def set_params(self, params: Dict[str, Any]):
        for key in params:
            if key not in self.__class__._default_params:
                raise ValueError(
                    f"Invalid param '{key}'. Valid params: {self.get_param_list()}"
                )
        self.params.update(params)
        self.prepare_input()

    @abstractmethod
    def prepare_input(self):
        raise NotImplementedError("prepare_input must be implemented in subclasses.")

    @abstractmethod
    def launch_kernel(self):
        raise NotImplementedError("launch_kernel must be implemented in subclasses.")

    def profile_kernel_us(
        self,
        stream: Optional[torch.cuda.Stream] = None,
        repeat=10,
        show_table=False,
        skip_warmup=False,
    ) -> float:
        assert self.inputs is not None, "Please initialize inputs first!"
        assert repeat > 0 and isinstance(repeat, int), "repeat must be an integer"
        if stream is None:
            stream = torch.cuda.Stream()

        # warmup
        if not skip_warmup:
            with torch.cuda.stream(stream):
                self.launch_kernel()
            torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with torch.cuda.stream(stream):
                for _ in range(repeat):
                    self.launch_kernel()

        events = prof.key_averages()
        max_cuda_time = max(
            (
                event.device_time
                for event in events
                if event.device_type != torch.autograd.DeviceType.CPU
                and event.key.startswith(self.__class__._key)
            ),
            default=0.0,
        )

        if show_table:
            print(events.table(sort_by="self_cuda_time_total", row_limit=-1))

        return max_cuda_time

    def get_best_performance_us(self) -> float:
        return self.profile_kernel_us()
