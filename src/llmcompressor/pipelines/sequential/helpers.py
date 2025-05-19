import contextlib
import inspect
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from accelerate.hooks import (
    ModelHook,
    add_hook_to_module,
    remove_hook_from_module,
    send_to_device,
    set_module_tensor_to_device,
)
from compressed_tensors import has_offloaded_params
from compressed_tensors.quantization import find_name_or_class_matches
from loguru import logger
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import PythonCode
from torch.fx.proxy import Argument
from torch.nn import Module
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.fx import HFTracer

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.helpers import calibration_forward_context, patch_attr
from llmcompressor.utils.pytorch.module import get_no_split_params

from .ast_helpers import autowrap_forwards

__all__ = ["trace_subgraphs", "Subgraph", "get_targets_from_modifiers"]


@dataclass
class Subgraph:
    """
    Dataclass specifying an executable subgraph of a model graph

    :param graph: subgraph of model graph
    :param input_names: argument names of the compiled forward function
    :param consumed_names: argument names which are not used by any subsequent subgraphs
        and can therefore be deleted from the intermediates cache
    """

    graph: Graph
    input_names: Set[str]
    consumed_names: Set[str]
    modules: List[Module]
    _code: Optional[PythonCode] = None

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the operations within the subgraph

        :param \\*args: argument inputs to subgraph forward function
        :param \\**kwargs: keyword inputs to subgraph forward function
        :return keyword outputs of subgraph forward function (non-consumed variables):
        """
        if self._code is None:
            self._code = self.graph.python_code("self")
            exec(self._code.src, self._code.globals)

        forward_fn = self._code.globals.get("forward")

        try:
            outputs = forward_fn(*args, **kwargs)
        except Exception as exception:
            raise RuntimeError(
                "Raised an exception during execution of the following code:\n"
                f"```\n{add_line_numbers(self._code.src)}\n```\n"
                "This is likely due to a violation of shape assumptions made when "
                "tracing"
            ) from exception

        return outputs


def trace_subgraphs(
    model: PreTrainedModel,
    sample_input: Dict[str, Any],
    sequential_targets: List[str],
    ignore: List[str],
) -> List[Subgraph]:
    """
    Trace a model to produce subgraphs, where each sequential target belongs to exactly
    one subgraph and where executing each subgraph in order is equivalent to executing
    the original model

    :param model: model being traced
    :param sample_input: inputs whose values will change during execution but whose
        __len__, __bool__, and __contains__ values are assumed constant across batches
    :param sequential_targets: list of patterns matching sequential targets
    :param ignore: modules to ignore during tracing, in the future will specify
        functions and methods to skip during tracing
    :return: a list of Subgraphs in order of execution
    """
    # find modules
    ignore = ["_update_causal_mask"]
    targets = match_modules(model, sequential_targets)
    ancestors = get_sequential_ancestors(model, targets)

    # initialize arguments
    tracer = get_tracer(model, ancestors)
    concrete_args = populate_concrete_args(model, sample_input)

    with contextlib.ExitStack() as stack:
        # calibration context
        stack.enter_context(calibration_forward_context(model))
        stack.enter_context(HooksMixin.disable_hooks())

        # flags useful for tracing
        stack.enter_context(patch_attr(model.config, "_attn_implementation", "eager"))
        stack.enter_context(patch_attr(torch.compiler, "_is_compiling_flag", True))

        # autowrap forwards
        stack.enter_context(autowrap_forwards(ancestors, ignore))
        stack.enter_context(patch_attr(type(model), "forward", model.forward.__func__))

        graph = GraphModule(
            model,
            tracer.trace(
                model,
                dummy_inputs=sample_input,
                concrete_args=concrete_args,
                complete_concrete_args_with_inputs_not_in_dummy_inputs=False,
                # bug in trace throws an error for variadic
                # args and kwargs in function signature
            ),
        )

    # copy metadata
    graph.config = model.config
    graph.class_for_deserialization = model.__class__
    graph.device = model.device

    # perform subgraph partition
    partitions = topological_partition(graph, targets)
    subgraphs = partition_graph(model, partitions, graph)
    trace_consumed_names(subgraphs)

    if len(subgraphs) != len(targets) + 1:
        logger.warning(
            f"Expected {len(targets)} subgraphs, but only traced {len(subgraphs)}. "
            "This is likely due to having wrapped code which calls sequential targets"
        )

    return subgraphs


def get_tracer(model: Module, ancestors: Set[Module]) -> HFTracer:
    """
    Get a tracer specialized for the given model. The resulting tracer will not trace
    inside of sequential targets, nor any modules which are not call graph ancestors of
    sequential targets

    Tracing within sequential targets is unnecessary, and tracing within offloaded
    modules may result in meta tensors being added to the model graph

    :param model: model being traced
    :param ancestors: modules which are ancestors of sequential targets
    :param ignore: modules to ignore during tracing, in the future will specify
        functions and methods to skip during tracing TODO
    """
    # check unlikely case that ancestors have direct params which are offloaded
    offloaded_modules = set(m for m in model.modules() if has_offloaded_params(m))
    offloaded_ancestors = offloaded_modules & ancestors
    if offloaded_ancestors:
        names = set(module.__class__.__name__ for module in offloaded_ancestors)
        logger.warning(
            "The following modules are call graph ancestors of sequential targets,"
            f"but also contain offloaded modules: {names}.\n"
            "These modules will not be traced, and any sequential target children will "
            "be executed jointly, which may lead to OOM errors"
        )

    class SequentialTracer(HFTracer):
        def create_arg(self, a: Any) -> Argument:
            # special extension allows models which depend on config values to be traced
            if isinstance(a, PretrainedConfig):
                kwargs = {k: self.create_arg(v) for k, v in a.to_dict().items()}
                return self.create_node("call_function", a.__class__, (), kwargs)

            else:
                return super().create_arg(a)

        def is_leaf_module(self, module: Module, module_qualified_name: str) -> bool:
            # TODO: cleanup
            nonlocal ancestors
            return module not in ancestors or module in offloaded_modules

    return SequentialTracer()


def populate_concrete_args(model: Module, sample_input: Dict) -> Dict:
    """
    Creates concrete args which, unlike the equivalent function provided by
    transformers.utils.fx, creates default values for variadic arguments, which are
    needed by some models.

    :param model: model being traced
    :param sample_input: values used to symbolically trace the model. All arguments
        to the model.forward function which are not in the sample_input are considered
        concrete args
    :return: dictionary mapping concrete argument names to their default values
    """
    sig = inspect.signature(model.forward)

    concrete_args = {}
    for parameter in sig.parameters.values():
        if parameter.name in sample_input:
            continue
        if parameter.kind == inspect._ParameterKind.VAR_POSITIONAL:
            value = list()
        elif parameter.kind == inspect._ParameterKind.VAR_KEYWORD:
            value = dict()
        elif parameter.name == "use_cache":
            value = False
        else:
            value = parameter.default

        concrete_args[parameter.name] = value

    return concrete_args


def match_node_to_target(graph: GraphModule, node: Node, targets: Set[Module]) -> Optional[Module]:
    if node.op != "call_module":
        return None
    
    node_module = graph.get_submodule(node.target)
    for target in targets:
        if isinstance(node_module, target):
            return target
        
    return None



def topological_partition(graph: GraphModule, targets: Set[Module]) -> List[List[Node]]:
    """
    Partition the graph into partitions such that each `target` belongs to exactly one
    partition and executing each partition depends only on intermediate values produced
    by executing the partitions before it.

    :param graph: graph being partitioned
    :param targets: target modules which will be assigned to disjoint partitions
    :return: list of partitions, where each partition is a list of nodes belonging to
        that partition
    """
    assert graph_is_well_formed(graph.graph)

    partitions: List[List[Node]] = [[]]
    remaining_indegrees = {
        node: len([node for node in node.all_input_nodes if node.op != "get_attr"])
        for node in graph.graph.nodes
    }
    partition_index = 0  # global counter

    targets_counter = {target: 0 for target in targets}
    targets_max = {target: 1 for target in targets}
    # HARD CODE
    from transformers.models.llama4.modeling_llama4 import Llama4TextMLP
    targets_max[Llama4TextMLP] = 64

    # start with graph input nodes,
    # but delay the `get_attr` nodes as long as possible
    queue = deque(
        node
        for node in graph.graph.nodes
        if remaining_indegrees[node] == 0 and node.op != "get_attr"
    )
    while len(queue) > 0:
        node = queue.popleft()

        # assign to partition
        partitions[partition_index].append(node)

        # guarantee targets are assigned to disjoint partitions
        matched = match_node_to_target(graph, node, targets)
        if matched is not None:
            targets_counter[matched] += 1
            
            if targets_counter[matched] >= targets_max[matched]:
                partition_index += 1
                partitions.append([])
                targets_counter[matched] = 0

        # recurse on last indegree only in order to guarantee that
        # the node is assigned to maximal partition
        for user in node.users:
            remaining_indegrees[user] -= 1
            if remaining_indegrees[user] == 0:
                queue.append(user)

    # an ideal implementation would involve implicitly consolidating partition indices
    # so that each node is assigned to the maximum partition possible (in order to delay
    # execution as long as possible), but saving these nodes for last covers the most
    # common and costly case (get_attr)
    for node in graph.graph.find_nodes(op="get_attr"):
        user_partitions = []
        for user in node.users:
            for index in range(len(partitions)):
                if user in partitions[index]:
                    user_partitions.append(index)
                    break
        partition_index = min(user_partitions)
        partitions[partition_index].insert(0, node)

    assert set().union(*partitions) == set(graph.graph.nodes)
    return partitions


def partition_graph(
    model: Module, partitions: List[List[Node]], parent_graph: GraphModule
) -> List[Subgraph]:
    """
    Convert each partition into a Subgraph. Each Subgraph returns a dictionary mapping
    of output node names to their computed values. Note that the `consumed_names`
    attribute of each Subgraph remains empty, to be later populated by
    `trace_consumed_names`

    :param model: model which owns the produced Subgraphs
    :param partitions: list of partitions, where each partition is a list of nodes
        belonging to that partition
    :return: list of subgraphs in order of execution
    """
    subgraphs = []

    # create subgraphs
    for partition_nodes in partitions:
        # create a new graph for the partition
        graph = Graph(model)
        node_map = {}

        # add placeholders for inputs not in this subgraph. use set to deduplicate
        new_input_nodes = {
            input_node
            for node in partition_nodes
            for input_node in node.all_input_nodes
            if input_node not in partition_nodes and input_node.op
        }
        for input_node in new_input_nodes:
            node_map[input_node] = graph.placeholder(input_node.name)

        # add the nodes to subgraph
        for node in partition_nodes:
            node_map[node] = graph.node_copy(node, lambda n: node_map[n])

        # add an output node to collect all subgraph outputs into a dictionary
        if len(graph.find_nodes(op="output")) <= 0:
            output_dict = {
                node.name: node_map[node]
                for node in partition_nodes
                if any(user not in partition_nodes for user in node.users.keys())
            }
            graph.output(output_dict)

        # save the subgraph for this partition
        graph.lint()
        input_names = set(node.name for node in graph.nodes if node.op == "placeholder")
        modules = get_subgraph_modules(graph, parent_graph)
        subgraphs.append(
            Subgraph(
                graph=graph,
                input_names=input_names,
                consumed_names=set(),  # populated later
                modules=modules,
            )
        )

        assert graph_is_well_formed(graph)

    return subgraphs


def trace_consumed_names(subgraphs: List[Subgraph]):
    """
    Populate the `consumed_names` attribute of each Subgraph according to when inputs
    are last used in order to vacate the `intermediates` cache and save memory

    :param subgraphs: list of subgraphs with empty `consumed_names` attributes
    """
    # populate consumed_names according to when inputs are last used
    # in order to vacate the `intermediates` cache and save memory
    all_input_names = set().union(*(subgraph.input_names for subgraph in subgraphs))
    for input_name in all_input_names:
        for subgraph in reversed(subgraphs):
            if input_name in subgraph.input_names:
                subgraph.consumed_names.add(input_name)
                break
        else:
            raise ValueError(f"Could not find input name {input_name} in subgraphs")


def graph_is_well_formed(graph: Graph) -> bool:
    """
    A graph is well formed if and only if
    `nodeA in NodeB.users <=> nodeB in Node.A.all_input_nodes`

    :param graph: graph being checked
    :return: True if the graph is well formed, False otherwise
    """
    for node in graph.nodes:
        for user in node.users:
            if node not in user.all_input_nodes:
                return False

        for input_node in node.all_input_nodes:
            if node not in input_node.users:
                return False

        if len(node.users) != len(set(node.users)) or len(node.all_input_nodes) != len(
            set(node.all_input_nodes)
        ):
            return False

    return True


def match_modules(model: Module, target_names: List[str]) -> Set[Module]:
    """
    Find modules whose names match the patterns given by `target_names`

    :param model: model containing submodules to find
    :param target_names: target patterns to find
    :return: all submodules matching `target_names`
    """
    return set(
        module
        for name, module in model.named_modules()
        if find_name_or_class_matches(name, module, target_names)
    )


def get_targets_from_modifiers(
    modifiers: List[Modifier], model: PreTrainedModel
) -> Tuple[List[str], List[str]]:
    """
    Infer sequential targets and ignore list from modifiers list

    :param model: model being calibrated
    :param modifiers: list of modifiers being applied during calibration
    :return: list of sequential targets and list of modules to ignore for tracing
    """
    # avoid circular import
    from llmcompressor.pipelines.registry import SEQUENTIAL_MODIFIERS

    sequential_modifiers = [
        modifier for modifier in modifiers if isinstance(modifier, SEQUENTIAL_MODIFIERS)
    ]

    if len(sequential_modifiers) >= 2:
        types = [type(modifier) for modifier in sequential_modifiers]
        logger.warning(
            "Cannot infer sequential targets from multiple sequential modifiers "
            f"({types}). Defaulting to {types[0]}"
        )
    elif len(sequential_modifiers) <= 0:
        types = [type(modifier) for modifier in modifiers]
        raise ValueError(f"Cannot infer sequential targets from list of {types}")

    modifier = sequential_modifiers[0]

    # infer sequential targets
    if modifier.sequential_targets is None:
        sequential_targets = get_no_split_params(model)
    elif isinstance(modifier.sequential_targets, str):
        sequential_targets = [modifier.sequential_targets]
    else:
        sequential_targets = modifier.sequential_targets

    return sequential_targets, modifier.ignore


def add_line_numbers(text: str) -> str:
    lines = text.splitlines()
    numbered_lines = [f"{i + 1} {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def get_sequential_ancestors(model: Module, targets: Set[Module]) -> Set[Module]:
    """
    Find modules which are call graph ancestors of the given sequential targets

    :param model: model containing sequential targets
    :param targets: sequential targets to find ancestors of
    :return: call graph ancestors of sequential targets
    """
    ancestors = set()

    def is_ancestor(module: Module) -> bool:
        if module in ancestors or module in targets:
            return True

        # eagerly compute list in order to avoid early stopping and :. missing ancestors
        _is_ancestor = any([is_ancestor(child) for child in module.children()])
        if _is_ancestor:
            ancestors.add(module)

        return _is_ancestor

    is_ancestor(model)
    return ancestors


def get_subgraph_modules(subgraph: Graph, parent_graph: GraphModule) -> List[Module]:
    """
    Get all submodules executed by `subgraph`
    :param subgraph: subgraph of parent_graph
    :param parent_graph: GraphModule describing the model,
        used for `get_submodule` method
    :return: all submodules executed by subgraph
    """
    modules_ops: List[Node] = subgraph.find_nodes(op="call_module")
    called_modules = [parent_graph.get_submodule(op.target) for op in modules_ops]
    return list({m for module in called_modules for m in module.modules()})


def infer_oneshot_device(oneshot_device: Optional[torch.device]) -> torch.device:
    if oneshot_device is None:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    else:
        return torch.device(oneshot_device)


def set_execution_device(model: Module, oneshot_device: Union[str, None]) -> Module:
    execution_device = infer_oneshot_device(oneshot_device)

    remove_hook_from_module(model, recurse=True)
    attach_execution_device_hook(model, execution_device)
    return model


def attach_execution_device_hook(
    module: Module,
    execution_device: torch.device,
):
    if len(list(module.parameters(recurse=False))) > 0:
        hook = AlignExecutionDeviceHook(execution_device=execution_device)
        add_hook_to_module(module, hook)

    for submodule in module.children():
        attach_execution_device_hook(submodule, execution_device)


# some submodules are called outside of the fx graph. Because these submodules do not
# appear in the graph, there is no way to know if they appear in the subgraph, and
# therefore no way to directly control their onloading behavior. Instead we must rely on
# a hook applied all offloadable modules and global variables to control them
offloading_disabled = False
disabled_offloading = set()


class AlignExecutionDeviceHook(ModelHook):
    def __init__(
        self,
        execution_device: Optional[Union[int, str, torch.device]] = None,
        skip_keys: Optional[Union[str, List[str]]] = None,
    ):
        self.execution_device = execution_device
        self.skip_keys = skip_keys

        self.devices = {}

    def init_hook(self, module: Module) -> Module:
        self.devices = {
            name: param.device for name, param in module.named_parameters(recurse=False)
        }
        return module

    def pre_forward(self, module, *args, **kwargs):
        for name in self.devices:
            set_module_tensor_to_device(module, name, self.execution_device)

        return send_to_device(args, self.execution_device), send_to_device(
            kwargs, self.execution_device, skip_keys=self.skip_keys
        )

    def post_forward(self, module, output):
        global offloading_disabled
        global disabled_offloading

        if not offloading_disabled:
            for name, device in self.devices.items():
                set_module_tensor_to_device(module, name, device)
        else:
            disabled_offloading.add((self, module))

        return output


@contextlib.contextmanager
def disable_onloading():
    global offloading_disabled
    global disabled_offloading
    offloading_disabled = True

    yield

    offloading_disabled = False
    for hook, module in disabled_offloading:
        hook.post_forward(module, None)

    disabled_offloading = set()
