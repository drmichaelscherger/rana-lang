# Rana Computational Model Specification

**Version:** 0.1  
**Status:** Initial draft for discussion  
**Date:** November 18, 2026

## 1. Purpose and scope

Rana expresses computation over arrays. This specification defines the logical values and operations that give a Rana computation meaning, independently of source syntax, compiler design, storage layout, and execution hardware.

Arrays are the primary computational abstraction, following the established design direction. The remaining rules below are proposed semantics for review, including the treatment of masks as first-class values. Within this draft, “must” identifies a requirement of the proposed model; it does not imply that the design has already been ratified.

Examples use descriptive mathematical notation, not proposed Rana syntax. Version 0.1 describes a finite, dense array core. Sparse arrays, distributed execution, automatic differentiation, and low-level hardware facilities are outside its scope.

## 2. Values, shapes, and types

An array value consists of an element type, a shape, and one element at each valid index. Its shape is an ordered tuple of nonnegative integer dimensions. Its rank is the number of dimensions, and its size is their product.

A scalar is a rank-zero array with shape `()` and exactly one element. A shape containing a zero dimension describes an empty array. A one-dimensional array is a logical sequence of values, regardless of how many elements a processor can execute together.

The initial element categories are Boolean, integer, and floating point. Each concrete numeric type must define its representation and arithmetic behavior. Numeric widths, implicit conversions, overflow, and exceptional floating-point behavior remain open in this draft. No portable result may depend on rules that have not been specified.

Shape is part of an operation's validity conditions whether dimensions are known before execution or discovered during execution. This model does not require dimensions to be part of a static type system.

## 3. Elementwise computation

An elementwise operation applies a scalar operation independently at each index of its result. Arithmetic produces numeric arrays; comparisons produce Boolean arrays.

Operands must have identical shapes, except that a scalar may be extended to every index of an array operand. Other implicit broadcasting is outside the version 0.1 core. Incompatible shapes must be reported as an error before the operation produces an observable update.

For example, adding the scalar `10` to `[1, 2, 3]` yields `[11, 12, 13]`. Comparing `[1, -2, 3]` with zero yields `[true, false, true]`.

Array truth is never implicit. A Boolean array must be reduced explicitly when a single Boolean decision is required.

## 4. First-class masks

A mask is a Boolean array used to identify participating elements. It can be stored, passed to a function, returned, and combined independently of the data it controls. The model requires no separate hardware mask type.

Logical AND, OR, and NOT operate elementwise. The reductions `any` and `all` produce Boolean results. Over an empty collection, `any` is false and `all` is true.

Two distinct operations use masks:

- **Selection:** Given a mask and two already-defined values of compatible shapes, selection chooses the first value at true positions and the second at false positions. Selection does not itself suppress evaluation of expressions used to produce those values.
- **Masked evaluation:** An operation evaluates only at active positions. Inactive positions perform no element computation, memory access, or observable effect. A value-producing masked operation must specify the values retained or supplied at inactive positions; a masked update leaves them unchanged.

These guarantees are semantic. An implementation may execute speculative work only when it cannot introduce an observable effect or error at an inactive position.

For example, computing `1 / x` under the mask `x != 0`, with zero as the inactive result, avoids division at zero positions. Selecting between an already-computed `1 / x` and zero does not provide that guarantee.

## 5. Indexing and structural operations

Indices are zero-based. A valid index contains one coordinate per dimension, with each coordinate between zero inclusive and the corresponding dimension exclusive. An active out-of-bounds access is an error. Negative indexing and implicit index wrapping are not part of the core.

Indexing selects elements; slicing selects a structured subset; gathering selects elements using explicit index arrays. Their logical results are values. Whether an implementation copies storage or shares it is not observable through these operations alone.

Reshape changes shape while preserving size and logical element order. Transposition permutes axes and their associated coordinates. For these definitions, logical element order varies the last coordinate fastest; this does not prescribe physical memory layout.

Boolean compaction is distinct from masked selection: it returns a one-dimensional array of active elements in logical order, so its result size may depend on data. Masked selection preserves shape.

## 6. Reductions

A reduction combines elements along explicitly identified axes using a specified combining operation. Reduced axes are removed from the result shape; reducing every axis produces a scalar.

A reduction must define its identity or initial value and its ordering contract. An empty reduction with neither an identity nor an initial value is an error. For example, Boolean `any` and `all` have the identities defined above; minimum and maximum require an explicit empty-input policy.

Reassociation is permitted only when it preserves the operation's specified behavior or is explicitly allowed by its numerical contract. Floating-point addition must not be assumed associative. Exact reproducibility policies and optional relaxed numerical modes remain open.

## 7. State, updates, and effects

Array values have value semantics: computing a new value does not change an existing value. Mutable state, where provided, is represented by explicit storage locations containing array values. The eventual syntax and ownership system are outside this specification.

An update reads its inputs from the state before that update and then applies its writes. Consequently, overlapping source and destination regions cannot change the result through execution order. A masked update changes only active destinations.

An update that writes the same destination more than once is an error unless the operation explicitly defines a conflict rule, such as accumulation with a specified combining operation. Ordinary scatter must not silently choose a winning writer.

Pure functions map input values to output values without observable effects. Functions that access mutable state or external resources must expose that behavior in their semantic contract. Effectful operations have an explicit sequencing relationship; array traversal order does not implicitly establish one.

## 8. Execution independence

Logical array shape and physical execution width are independent. A conforming implementation may use scalar instructions, SIMD, threads, accelerators, or combinations of these, provided it preserves the specified values, active-element behavior, effects, and numerical contracts.

Storage layout, tiling, fusion, temporary elimination, and evaluation timing are implementation choices subject to the same requirement. Optimizations must respect state dependencies. No program may infer a hardware register width from the shape of an ordinary array.

An optional future facility may expose hardware-specific operations through a separate contract. Such a facility does not redefine ordinary array semantics.

## 9. Errors and unresolved decisions

Shape mismatches, invalid active indices, and conflicting writes are semantic errors, rather than permission for arbitrary behavior. Detection may occur before or during execution. An individual failed array operation must not expose partial updates; this does not imply rollback of earlier completed operations. Error reporting and recovery mechanisms remain open.

Before implementing the Python reference library, the next revision should resolve:

1. Concrete numeric types, conversions, overflow, division, and exceptional values.
2. Whether broadcasting beyond scalar extension belongs in the core.
3. The interface for masked evaluation and inactive result values.
4. Reduction ordering, floating-point reproducibility, and relaxed modes.
5. The initial scope of mutable storage, aliasing, and effectful functions.

The Python library should serve as an executable reference for agreed semantics. Python conventions and the behavior of any supporting array library must not silently become Rana language rules.
