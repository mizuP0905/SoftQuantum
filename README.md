# SoftQuantum

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**SoftQuantum** is a local quantum circuit simulator written in Python. It provides a NumPy CPU backend, an optional CuPy GPU backend, and a desktop QASM editor for running and debugging OpenQASM-like programs.

Current local package metadata version: `0.4.0`

## Highlights

- CPU execution with optional NVIDIA GPU acceleration through CuPy
- Interactive GUI (`QASM Studio`) built with Tkinter
- OpenQASM-style parsing with support for headers, includes, gate declarations, control flow, and shots
- Large built-in gate set including standard, controlled, and research-oriented gates
- Debug-friendly commands such as `print_state`, `print_probs`, and `print_creg`
- Deterministic seeding for reproducible measurements, including after `qreg` resets

## What Changed In This Update

- The default GPU path now uses **CuPy** instead of the older `_svcuda` compiled extension workflow.
- The simulator now performs an explicit CuPy runtime self-check, including simple gate-like operations, and reports backend availability clearly.
- Silent GPU fallback was removed. If the CuPy backend is active and fails during execution, the simulator raises an error instead of quietly switching to CPU.
- `qreg` reinitialization now preserves the configured random seed.
- OpenQASM-style language support was expanded with:
  - `OPENQASM` headers
  - `include`
  - `gate` declarations with parameters
  - `qubit[]` / `bit[]`
  - `for`, `while`, `if`, `else`, `if (classical)`
  - `shots`
  - register measurement such as `measure q -> c`
- GUI keyword highlighting and sample programs were updated to match the new language support.
- `reset` now follows measurement-based circuit reset semantics.
- `barrier` and `delay[...]` instructions with operands are accepted as no-ops.
- **Breaking change:** multi-qubit matrices now use MSB-first target ordering. `apply_unitary([q0, q1], U)` interprets `U` in the basis `|q0 q1> = |00>, |01>, |10>, |11>`.
- Added `DensityMatrixSimulator`, a CPU density-matrix backend with deterministic CPTP channel evolution.
- `execute_qasm(..., mode="density_matrix")` or `execute_qasm(..., density_matrix=True)` runs QASM programs on the density-matrix backend.
- Added an OpenQASM conformance suite and a standalone density-matrix benchmark script.
- Regression tests were added for the parser/runtime behavior and numerical audit cases.

## Requirements

- Python 3.10 or later
- `numpy`
- `tkinter` for the GUI (usually included with standard Python installations)
- Optional: `pytest` for running the test suite
- Optional GPU backend:
  - NVIDIA GPU
  - compatible NVIDIA driver / CUDA runtime
  - a matching CuPy package

## Installation

### CPU-only setup

```bash
pip install numpy
pip install .
```

### Optional GPU backend

The project currently exposes a CuPy extra in `setup.py`:

```bash
pip install ".[cuda]"
```

This currently installs:

```bash
pip install "cupy-cuda12x>=13.0"
```

No separate `python setup.py build_ext --inplace` step is required for the current default runtime.

## GPU Backend Notes

- On import, SoftQuantum runs a small CuPy self-check.
- If CuPy cannot be imported or initialized, the simulator reports the GPU backend as unavailable and continues on the NumPy backend.
- If the CuPy backend is active and a GPU operation fails during execution, an explicit exception is raised.
- The legacy `cuda_statevector.cu` source is still kept in the repository for reference and experimentation, but it is not required for normal installation or runtime use.
- GPU availability can still depend on local environment details such as driver state, CUDA compatibility, and whether CuPy can access its runtime cache / temporary compilation paths.
- On Windows, CuPy/NVRTC may need writable temporary and cache directories. If backend initialization or execution reports a permission problem, set `TMP`, `TEMP`, and optionally `CUPY_CACHE_DIR` to writable directories before launching Python.

## Simulation Conventions

- Qubit `q=0` is the least significant bit (LSB) of the statevector basis index. For example, `X(0)|00>` produces basis index `1`.
- Multi-qubit matrix ordering is now MSB-first by target-list order. For `apply_unitary([q0, q1], U)`, `U` is interpreted in the basis `|q0 q1> = |00>, |01>, |10>, |11>`. For three or more targets, the list is also interpreted MSB-to-LSB.
- This is a breaking change for callers that passed custom 4x4 or 8x8 matrices using the older local `targets[0]` LSB convention. Reorder those matrices before passing them to the current API.
- QASM operands follow the same rule: `cx q[0], q[1]` means control `q[0]`, target `q[1]`, and custom multi-qubit matrix commands use the operand list as MSB-to-LSB.
- `measure_all()` returns a list indexed by qubit number, so result element `0` is qubit `q[0]`. Printed bitstrings and probability labels use the integer basis index with `q[0]` as the LSB.
- Statevector `reset(q)` is a trajectory reset: it measures `q` without writing to `creg`, then applies `X(q)` if the sampled outcome was `1`.
- Density-matrix `reset(q)` is a deterministic CPTP reset channel using Kraus operators `|0><0|` and `|0><1|`.
- `QuantumSimulator.apply_channel` keeps the statevector trajectory behavior for backwards compatibility: it samples one Kraus branch.
- `DensityMatrixSimulator.apply_channel` applies the full density-matrix channel `rho -> sum_i K_i rho K_i^dagger` and is deterministic except for later measurement sampling.
- The density-matrix GPU path is not implemented yet. `DensityMatrixSimulator(prefer_gpu=True)` still reports and uses the CPU density backend.

## Supported QASM Features

- Legacy commands such as `qreg`, `creg`, `measure`, `reset`, `seed`, `print_state`, `print_probs`, `print_creg`
- OpenQASM-style declarations such as `OPENQASM 3`, `qubit[n]`, `bit[n]`
- Built-in includes: `stdgates.inc`, `qelib1.inc`
- Local relative includes such as `include "custom.inc"`
- User-defined gates:

```qasm
gate rot(theta) a { rx(theta) a; }
```

- Classical control flow:

```qasm
if (c == 1) { x q[0]; } else { h q[0]; }
while (c == 1) { x q[0]; measure q[0] -> c[0]; }
for int i in [0:1] { x q[i]; }
```

- Multi-shot execution:

```qasm
shots 1024;
```

- Timing/layout no-ops:

```qasm
barrier q[0], q[1];
delay[10ns] q;
```

### Density-Matrix QASM Mode

Noise-free circuits should produce matching probabilities in statevector and density-matrix mode. Noisy circuits in density-matrix mode apply deterministic channels before any measurement sampling:

```python
from quantum_simulator_global import DensityMatrixSimulator, execute_qasm

sim = DensityMatrixSimulator(1, seed=42)
result = execute_qasm(
    sim,
    lines=[
        "OPENQASM 3",
        "qubit[1] q",
        "h q[0]",
        "noise_phaseflip q[0] 0.5",
        "print_probs",
    ],
    mode="density_matrix",
)
print(result["rho"])
print(result["probabilities"])
```

## Gate Library

SoftQuantum includes standard and extended gates such as:

- `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`
- `sx`, `sxdg`, `p`, `u`, `u1`, `u2`, `u3`
- `rx`, `ry`, `rz`
- `swap`, `iswap`, `iswap_theta`, `iswap_pow`, `iswapdg`
- `fsim`, `syc`, `phased_iswap`, `phasedfsim`, `cz_wave`
- `rxx`, `ryy`, `rzz`
- `cx`, `cy`, `cz`, `ch`, `cs`, `ct`, `cp`, `crx`, `cry`, `crz`, `cu1`, `cu3`
- `ccx` / `toffoli`, `cswap`
- several noise commands including bit-flip, phase-flip, depolarizing, amplitude damping, and phase damping helpers

## Running QASM Studio

Launch the GUI from the project directory:

```bash
python qasm_gui.py
```

The editor includes:

- syntax highlighting for QASM keywords, numbers, and comments
- built-in sample programs
- statevector / density-matrix mode selection
- backend-aware execution output
- status reporting for the detected CPU / GPU backend

## Python API Example

```python
from quantum_simulator_global import DensityMatrixSimulator, QuantumSimulator, execute_qasm

program = [
    "OPENQASM 3",
    'include "stdgates.inc"',
    "qubit[2] q",
    "bit[2] c",
    "h q[0]",
    "cx q[0], q[1]",
    "measure q -> c",
]

sim = QuantumSimulator(2, seed=42)
result = execute_qasm(sim, lines=program)
print(result["backend"])
print(result["creg"])

dm_sim = DensityMatrixSimulator(2, seed=42)
dm_result = execute_qasm(dm_sim, lines=program, mode="density_matrix")
print(dm_result["rho"])
```

Density-matrix simulation stores an `N x N` matrix for `N = 2**num_qubits`, so memory grows as `O(4^n)`. Use the statevector backend for larger noise-free circuits. See `docs/performance.md` and `docs/density_matrix_gpu_design.md` for benchmark guidance and the planned GPU path.

## Example Program

```qasm
OPENQASM 3;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

gate bell(a, b) {
    h a;
    cx a, b;
}

bell(q[0], q[1]);
measure q -> c;
print_creg;
```

## Tests

Install `pytest` if needed:

```bash
pip install pytest
python -m pytest tests/test_quantum_simulator.py
```

OpenQASM conformance coverage:

```bash
python -m pytest tests/test_openqasm_conformance.py -q
```

Density benchmark smoke test:

```bash
python benchmarks/benchmark_density_matrix.py --max-qubits 5 --repeat 1 --output-dir benchmarks/results
```

Recent regression coverage includes:

- seed preservation across `qreg` resets
- user-defined gates
- OpenQASM-style register measurement
- `for`, `while`, and classical `if/else`
- `shots`
- local `include`
- reset, bounds, Kraus/unitary validation, endian conventions, and CPU/GPU parity audit cases
- MSB-first target matrix ordering migration tests
- density-matrix unitary, channel, reset, measurement, QASM mode, and trace/Hermiticity/PSD checks
- OpenQASM support and negative conformance cases

## Documentation

- OpenQASM support matrix: `docs/openqasm_conformance.md`
- Density-matrix GPU design: `docs/density_matrix_gpu_design.md`
- Performance notes and benchmark usage: `docs/performance.md`

## Notes

- Relative `include` paths are resolved from the current source file location.
- Absolute include paths are intentionally rejected.
- The GUI and simulator live in simple module files: `qasm_gui.py` and `quantum_simulator_global.py`.

## License

This project is licensed under the **GNU AGPL v3**.
