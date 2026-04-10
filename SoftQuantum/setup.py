from setuptools import setup


setup(
    name="softquantum",
    version="0.3.0",
    py_modules=["quantum_simulator_global", "qasm_gui"],
    extras_require={
        "cuda": ["cupy-cuda12x>=13.0"],
    },
)
