from setuptools import setup


setup(
    name="softquantum",
    version="0.4.0",
    py_modules=["quantum_simulator_global", "qasm_gui"],
    python_requires=">=3.10",
    install_requires=["numpy"],
    extras_require={
        "cuda": ["cupy-cuda12x>=13.0"],
        "test": ["pytest"],
        "dev": ["pytest"],
    },
)
