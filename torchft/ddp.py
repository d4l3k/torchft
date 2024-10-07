import os
from typing import Optional, TYPE_CHECKING
import sys

from torch.nn import parallel
import torch
from torch import nn
from torch.distributed.algorithms.join import Joinable
import torch.distributed as dist
from torchft.process_group import ProcessGroup

if TYPE_CHECKING:
    from torchft.manager import Manager


class DistributedDataParallel(nn.Module):
    """
    A pure reimplementation of the DDP wrapper.
    """

    def __init__(self, manager: "Manager", module: nn.Module):
        super().__init__()

        self.module = module

        def post_grad_hook(p):
            if p.grad is not None:
                # TODO: use the torch reducer
                manager.allreduce_grad(p.grad)

        for p in module.parameters():
            p.register_post_accumulate_grad_hook(post_grad_hook)

    def forward(self, *args: object) -> object:
        return self.module(*args)


class HackedDistributedDataParallel(parallel.DistributedDataParallel):
    """
    This is a patched DistributedDataParallel implementation that makes it
    compatible with torchft.

    Important notes:
    * This requires states to be synced on step 0 using an external mechanism
      rather than an internal broadcast.
    * Using non-basic features of the DDP may cause your model to catch fire as
      they haven't been tested with torchft.
    * This doesn't any sanity checks such as verifying parameter sizes are the
      same across workers.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroup,
        bucket_cap_mb: Optional[int] = None,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False,
        mixed_precision: object = None,
    ) -> None:
        # Bypass normal DDP init
        nn.Module.__init__(self)
        Joinable.__init__(self)

        self.logger = None

        self.module = module
        self.process_group = process_group
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.mixed_precision = mixed_precision
        self.static_graph = False
        if self.mixed_precision is not None:
            logger.warning("Received mixed precision config %s", self.mixed_precision)

        self._delay_all_reduce_params = []
        self.parameters_to_ignore = set()
        self.device_ids = None
        self.output_device = None
        self.broadcast_buffers = False
        self._use_python_reducer = False
        self._accum_grad_hooks = []
        self._lazy_init_ran = False
        self._delay_all_reduce_all_params = False
        self._delay_grad_buffer = None

        self._module_parameters = [
            p
            for n, p in module.named_parameters()
            if n not in self.parameters_to_ignore
        ]
        if not any(p.requires_grad for p in self._module_parameters):
            self._log_and_throw(
                RuntimeError,
                "DistributedDataParallel is not needed when a module "
                "doesn't have any parameter that requires a gradient.",
            )

        # Check that a module does not have Uninitialized parameters
        for param in self._module_parameters:
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                self._log_and_throw(
                    RuntimeError,
                    "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )

        # reduction bucket size
        if bucket_cap_mb is None:
            # default case (bucket cap is 25 MiB)
            bucket_cap_mb = 25
            self.bucket_bytes_cap_default = True
        else:
            self.bucket_bytes_cap_default = False
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )

        # Build parameters for reducer.
        parameters, expect_sparse_gradient = self._build_params_for_reducer()

        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)

        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            static_graph=False,
        )

    def _ddp_init_helper(
        self,
        parameters,
        expect_sparse_gradient,
        param_to_name_mapping,
        static_graph,
    ):
        """
        DDP init helper function to manage parameters, grad hooks, logging, and SyncBatchNorm.

        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        # Notice, the parameters order is not in the order in which they are used,
        # especially in models with control flow.
        #
        # Alongside parameters are not presented in the real execution order,
        # if a certain model happens to also
        #   1) have other collectives comm ops in its backward graph.
        #   2) have unused parameter in subset ranks of the whole world.
        # bucketing could insert ALL-REDUCE comm op too early on the rank with unused parameter,
        # matching up with other collectives comm ops on other ranks unexpectedly.
        #
        # In order to handle this corner case, when the parameters are not in the real execution order,
        # we don't do bucketing, thus only one ALL-REDUCE is inserted after all the gradients
        # of the whole graph are computed.
        #
        # Notice, here we only disable bucketing for the first iteration.
        # After the first iteration, it's OK to rebuild buckets,
        # because "bucket rebuild" bucketizes parameters based on its real execution order in backward graph.

        # Can remove this branching once #73732 is landed.
        if static_graph is True or self.find_unused_parameters is False:
            bucket_size_limits = [sys.maxsize]
        else:
            if self.bucket_bytes_cap_default:
                bucket_size_limits = [
                    dist._DEFAULT_FIRST_BUCKET_BYTES,
                    self.bucket_bytes_cap,
                ]
            else:
                bucket_size_limits = [self.bucket_bytes_cap]
        (
            bucket_indices,
            per_bucket_size_limits,
        ) = dist._compute_bucket_assignment_by_size(
            parameters,
            bucket_size_limits,
            expect_sparse_gradient,
        )

        # Remember index for parameters if we are in mixed precision, as we
        # need to pass in index to Reducer's autograd hook via python.
        if self.mixed_precision is not None:
            for i, p in enumerate(parameters):
                p._idx = i

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            list(reversed(per_bucket_size_limits)),
            self.process_group,
            expect_sparse_gradient,
            # The bucket size limit is specified in the constructor.
            # Additionally, we allow for a single small bucket for parameters
            # that are defined first, such that their gradients don't spill into
            # a much larger bucket, adding unnecessary latency after gradient
            # computation finishes. Experiments showed 1MB is a reasonable value.
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            self.gradient_as_bucket_view,
            param_to_name_mapping,
            # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
            # bucket.
            (
                dist._DEFAULT_FIRST_BUCKET_BYTES
                if self.bucket_bytes_cap_default
                else self.bucket_bytes_cap
            ),
        )

        self.logger = dist.Logger(self.reducer)
        # Set as a weak reference to avoid reference cycle between
        # logger and reducer.
        self.reducer.set_logger(self.logger)

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self.module)
