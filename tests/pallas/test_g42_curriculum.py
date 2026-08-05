import ast

import pytest

from opjax.pallas.g42_curriculum import G42CurriculumError, mutate_kernel

SOURCE = '''from jax.experimental import pallas as pl

def _kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]

def workload(x):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(_kernel, out_shape=x, in_specs=(spec,), out_specs=spec)(x)
'''


@pytest.mark.parametrize(
    "mutation",
    ["reversed_blockspec", "illegal_block_geometry", "unsafe_grid_index", "incomplete_compute"],
)
def test_mutations_are_parseable_and_change_source(mutation: str) -> None:
    result = mutate_kernel(SOURCE, mutation)
    ast.parse(result)
    assert result != SOURCE


def test_unknown_mutation_fails_closed() -> None:
    with pytest.raises(G42CurriculumError, match="MUTATION_INVALID"):
        mutate_kernel(SOURCE, "unknown")
