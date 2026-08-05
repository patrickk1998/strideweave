<p align="center">
  <img src="assets/strideweave-mark.svg" alt="StrideWeave" width="140" height="140">
</p>

# StrideWeave

<p align="center">
  <a href="https://github.com/patrickk1998/strideweave/actions/workflows/ci.yml"><img src="https://github.com/patrickk1998/strideweave/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

StrideWeave is an experimental framework for tensor computation being developed by Patrick Krusiec. It is currently in a state of rapid development. I hope to release a somewhat stable version very soon. What makes StrideWeave different from other machine learning libraries? Well...

## Layouts, Layouts, Layouts!

Layouts are the mortar that binds everything in StrideWeave together. A layout is simply a shape and a stride that define a map from a coordinate space onto a flat index space. But unlike in PyTorch, layouts in StrideWeave are hierarchical. They have a CuTe-style layout algebra. You can compose them, divide them into tiles, or assemble tiles into a large layout. A flat index can also be expanded back into a coordinate of a multi-mode shape, allowing a layout to map onto a non-flat space.

## Tensors

We have layouts, and given this is a tensor computation library the next logical step is to define what a Tensor is. The obvious starting point is to define a Tensor as a datatype, a layout, and a pointer into a buffer in memory. But a problem quickly arises with this definition because of a hidden assumption: that every value of a datatype is self-contained. This was true in 2016: every FP32 value was 32 contiguous bits. This was true in 2019: every bf16 value was 16 contiguous bits. That has stopped being true today. An MXFP4 value is not just 4 contiguous bits; those 4 bits belong to a block of 32 values that share an 8-bit scaling factor. To store an MXFP4 tensor we need an array of 4-bit values, an array of 8-bit scales, and a way to connect them. Thankfully we have layouts.

A tensor in StrideWeave is an ordered set of sub-tensors. Each sub-tensor is composed of a carrier – we will talk about them in the section below, but they generalize the idea of a buffer in memory – and a layout from the coordinate space onto a flat index space. Between each sub-tensor in the ordered set, there is a layout that maps from the coordinate space of one sub-tensor to the coordinate space of the next. Remember, a flat index expands back into a coordinate of a multi-mode shape, so these adjacent layouts map onto a non-flat space.

## Moving Upwards: Carriers

The purpose of CuTe layouts, which the layout system in StrideWeave draws inspiration from, is to represent a hierarchical memory space. A tensor may exist in global memory, but it must be tiled, and those tiles must be assigned to CTAs across the GPU and moved into shared memory. Hierarchical memories don't stop at the level of global memory. There exist memory levels above the global memory stored in HBM. On a single machine: LPDDR5, and then NVMe block devices. And across machines: the total memory of a scale-up domain, such as a Blackwell NVL72 rack, and then the total memory of a cluster connected with a scale-out network.

The carrier seeks to extend the machinery that CuTe and CUTLASS use to represent data across a memory hierarchy on a GPU upward. It represents a flat index space of fixed size that stores values representable as a constant number of contiguous bits. The key idea is that the flat index space can be tiled and spread across different nodes, or across different memory levels on a single node. Operations on tensors are dispatched against the carrier.

## The Current State

I am still experimenting with the exact semantics of carriers and operations in StrideWeave. If you are interested in working on this, please feel free to contact Patrick Krusiec.
