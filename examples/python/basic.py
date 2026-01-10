#!/usr/bin/env python3
"""
PyFlame Basic Example

Demonstrates basic tensor operations and the lazy evaluation model.
"""

import pyflame as pf

print(f"PyFlame v{pf.__version__} Basic Example")
print("=" * 40)

# Create tensors
print("\nCreating tensors...")
a = pf.randn([3, 4])
b = pf.randn([3, 4])

print(f"a: {a}")
print(f"b: {b}")

# Arithmetic operations (lazy - not computed yet)
print("\nBuilding computation graph...")
c = a + b
d = c * 2.0
e = pf.relu(d)
f = e.sum()

print("Operations recorded. Graph not yet executed.")
print(f"f is lazy: {pf.is_lazy(f)}")

# Force evaluation
print("\nEvaluating...")
pf.eval(f)

print(f"f is lazy: {pf.is_lazy(f)}")
print(f"Result: {f.numpy()}")

# Access the computation graph
print("\nComputation graph:")
pf.print_graph(f)

# Graph statistics
graph = pf.get_graph(f)
print(f"\nGraph statistics:")
print(f"  Nodes: {graph.num_nodes()}")
print(f"  Operations: {graph.num_ops()}")
print(f"  Estimated memory: {graph.estimated_memory_bytes()} bytes")
