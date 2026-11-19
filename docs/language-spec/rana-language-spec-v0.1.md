# Rana Language Specification

**Version:** 0.1  
**Status:** Consolidated design draft  
**Date:** November 18, 2026

## 1. Status and notation

This document defines the Rana programming model and the semantic requirements for a conforming Rana implementation. Rana is a modern associative programming language derived from the ASC model. Its central abstraction is a single instruction stream operating collectively over a set of logical processing elements (PEs), with content-based search determining which PEs respond to subsequent operations.

The words **must**, **must not**, **should**, and **may** are normative. Code examples use provisional syntax: they illustrate required behavior but do not freeze the grammar. Items explicitly marked **Open** are not part of the portable v0.1 contract.

Rana separates three concerns:

1. the abstract associative machine defined here;
2. the source syntax and compiler representation; and
3. physical execution on scalar CPUs, SIMD units, GPUs, or other systems.

Observable behavior must not depend on the number of physical lanes, threads, or processors used by an implementation.

## 2. Design principles

Rana preserves the essential ASC model while replacing its historical syntax and implementation constraints.

- **Search, process, retrieve.** Programs find records by content, process the responders collectively, and retrieve or reduce associated values.
- **Associative memory is a logical abstraction.** It may be implemented by equally laid-out PE-local memories, ordinary arrays, SIMD vectors, GPU storage, or other representations. Dedicated CAM hardware is not required.
- **Responders are fundamental.** A responding set is part of execution state, not merely a temporary Boolean array.
- **One logical instruction stream is the v0.1 foundation.** Every collective statement denotes one operation issued to its active PEs. Multiple instruction streams are reserved for a later extension.
- **Masks are first-class values.** They may be stored, combined, passed, returned, and used to establish execution context.
- **Parallel results are deterministic unless an operation explicitly says otherwise.** Physical scheduling cannot select a winning writer or silently change numerical results.
- **Semantics precede performance.** An operation retains its meaning even when a target cannot implement it in constant time or in one physical instruction.

## 3. Abstract associative machine

### 3.1 Processing elements

A logical PE is an element of an associative space. It has:

- a stable identity within that space;
- one value for every field in the space's schema;
- an occupancy state; and
- participation state maintained by the current instruction stream.

Logical PEs need not correspond one-to-one with physical processing elements. An implementation may time-multiplex any number of logical PEs over any number of physical resources.

### 3.2 Associative spaces

An associative space is a finite collection of equally structured logical PEs. A space declares a schema of named fields. Conceptually, issuing a field access broadcasts one field address to the active PEs, and each PE accesses its own local value.

Example syntax:

```rana
space Employee(capacity: 1_000_000) {
    id: i64;
    age: i64;
    salary: f64;
    department: i64;
}
```

Each mask and distributed field value has a domain identifying its space. Values from different domains cannot be combined without an explicit mapping operation.

### 3.3 Occupancy

A space may be dense or dynamically occupied. Its occupancy mask identifies PEs that currently contain valid records. Unoccupied PEs never participate in ordinary searches or collective updates.

For a space `S`, its initial execution domain is `occupied_mask(S)`. Allocation activates an unoccupied PE and initializes every field before that PE becomes observable. Release removes a PE from occupancy and from every active or responder set derived from the space.

The choice of which free PE an unordered allocation uses is not portable. Programs requiring stable selection must supply or compute an explicit identity.

### 3.4 Instruction-stream state

For each entered space, the instruction stream maintains:

- the **occupied set**;
- the **active set**, whose PEs may execute the next collective operation; and
- the most recently established **responder set**.

A search predicate is evaluated only by PEs in the active set. The resulting responder set is the intersection of the active set, occupancy, and the predicate result.

Entering a nested execution context saves the enclosing active and responder sets. Leaving the context restores them, including on structured early exit or error propagation.

## 4. Values and types

### 4.1 Scalar types

The initial scalar types are:

- `bool`;
- signed 64-bit integer `i64`; and
- IEEE 754 binary64 floating point `f64`.

`bool` is not numeric. Mixed numeric operations require explicit conversion. Additional widths, characters, strings, and user-defined numeric types are outside the portable v0.1 core.

### 4.2 Numeric behavior

Ordinary `i64` arithmetic is checked. Overflow, including intermediate overflow, is an error. Integer quotient truncates toward zero and is distinct from remainder. `/` accepts `f64`; integer operands require explicit conversion. Division by zero is an error for both integer and floating-point operations.

Floating-point rounding is round-to-nearest, ties-to-even. `f64` supports infinities and NaNs. Conversion from `f64` to `i64` truncates toward zero and fails for NaN, infinity, or an out-of-range result.

**Open:** Complete NaN payload, signaling-NaN, overflow, underflow, and literal-typing rules remain to be specified. Programs whose result depends on an open rule are not portable v0.1 programs.

### 4.3 Distributed fields

A field expression evaluated in a space denotes one scalar value per PE in its domain. Elementwise arithmetic, comparison, and logical operations apply independently at corresponding PEs. A scalar operand is extended to every active PE. No other implicit broadcasting is permitted.

Array values may exist as field values or ordinary local values. Their shape and indexing rules follow section 9; arrays do not replace associative spaces or responder semantics.

### 4.4 Records and structures

Structures group named values. A space schema behaves as a distributed structure: each field is distributed across PEs while the fields at one PE form that PE's logical record.

### 4.5 Masks

A mask is a first-class Boolean value over a specific space domain. Masks support `and`, `or`, `xor`, and complement, plus `any`, `all`, and `count`. Complement is relative to the mask's occupied domain, not to unused capacity.

A mask may be created by comparison, copied, stored, passed to a function, returned, or combined with compatible masks. The compiler may avoid materializing it when doing so cannot change observable behavior.

### 4.6 PE handles and optional values

A PE handle identifies one occupied PE in one space and carries the generation needed to detect invalidation after release and reuse. Dereferencing an invalid handle is an error.

Operations that may find no responder return an optional result or require an explicit empty policy. Absence must not be represented by an arbitrary sentinel value.

## 5. Expressions and collective updates

### 5.1 Evaluation domain

A distributed expression is evaluated only at active PEs unless an operation explicitly names another compatible mask. Inactive PEs perform no controlled field access, arithmetic, memory access, or observable effect and cannot raise element-local errors for the suppressed computation.

Type and domain compatibility are checked even when the active set is empty.

### 5.2 Snapshot update semantics

A collective update reads all inputs from the state preceding that update, computes all active results, and commits only if the whole update succeeds. A failure exposes no partial writes. Earlier completed statements are not rolled back.

For example, swapping two fields with one parallel assignment observes both old values:

```rana
(Employee.x, Employee.y) = (Employee.y, Employee.x);
```

Inactive destinations remain unchanged.

### 5.3 Write conflicts

Writing one destination more than once in one collective update is an error unless the operation declares a conflict rule. Accumulating scatter and grouped reduction are defined conflict-resolving operations; ordinary scatter is not.

### 5.4 Selection and masked evaluation

Elementwise selection chooses between already computed values and does not suppress evaluation of either input. Masked evaluation controls the computation itself.

A value-producing masked operation must provide an explicit fallback for inactive positions. A masked update retains the old destination value. Passing a previously computed erroneous expression to a selection does not retroactively suppress its error.

## 6. Associative search and execution context

### 6.1 Search

Searching a space evaluates a predicate over the current active set and produces a responder mask.

```rana
let senior = search Employee where age >= 65;
```

The search does not by itself mutate fields. The result may later be reused regardless of changes to the current active context, subject to occupancy and handle validity.

### 6.2 `where`

A `where` statement establishes a nested active set equal to the enclosing active set intersected with its mask. Its body executes collectively for those responders. The enclosing execution context is restored afterward.

```rana
where Employee (age >= 65) {
    salary = salary * 1.05;
}
```

If the responder set is empty, the body performs no distributed computation. Scalar control effects inside the body must still obey their operation-specific rules.

### 6.3 Associative conditional

An associative conditional partitions the enclosing active set into true and false responder sets. In the single-stream v0.1 model, the true body executes first under the true set, then the false body executes under the complementary false set. Both are computed from the same saved enclosing set.

```rana
parallel if Employee (salary < 50_000.0) {
    band = 0;
} else {
    band = 1;
}
```

The order of branch execution is observable only through permitted scalar effects. A future multiple-instruction-stream extension may execute disjoint branches concurrently while preserving the defined data dependencies.

### 6.4 Scalar conditional

An ordinary scalar `if` requires a scalar `bool`. A mask has no implicit truth value; the program must use `any`, `all`, `none`, or another explicit reduction.

## 7. Associative iteration

### 7.1 Collective parallel loop

A collective parallel loop repeatedly executes its body for the current responder set. Its continuation condition is recomputed from the saved loop domain each iteration.

```rana
parallel while Employee (unprocessed) {
    // all currently responding employees execute this body
    unprocessed = false;
}
```

The loop terminates when no PE responds. Updates in one iteration are visible to the next. The body must not rely on physical PE traversal order.

### 7.2 Individual-responder iteration

Some algorithms require processing responders one at a time. An individual-responder loop snapshots its initial candidate set, selects one candidate according to an explicit selection policy, executes the body with only that PE active, removes it from the remaining snapshot, and repeats.

```rana
for responder e in Employee where selected order by id ascending {
    emit(e.id);
}
```

Portable policies include ascending stable identity and descending stable identity. An `unordered` policy permits any responder and therefore cannot be used when selection order affects a portable result.

New responders created after the loop snapshot are not visited. Released responders are skipped. Nested execution restores the one-responder context after it returns.

### 7.3 Early exits

`break` exits the nearest loop and restores its enclosing context. `continue` advances that loop according to its selection or collective-continuation rule. `return` restores all intervening contexts before returning from the function.

## 8. Associative operations

### 8.1 Scalar reductions

`count`, `any`, `all`, `sum`, `min`, and `max` reduce values over an explicitly identified responder set or the current active set. `any` of an empty set is `false`; `all` of an empty set is `true`; `count` is zero. Other reductions must define an identity, an initial value, or an explicit empty-result policy.

For each result, reduction traverses logical PE identity in ascending order. The accumulator starts from the identity or explicit initial value and applies `combine(accumulator, element)` sequentially. Implementations may optimize only when the specified result and errors remain identical. There is no relaxed v0.1 reduction mode.

### 8.2 Associated extrema

`argmin` and `argmax` return both an extreme value and a handle to a PE containing it. Ties are resolved by the lowest stable PE identity unless another explicit tie policy is supplied. Empty input returns none.

Associated retrieval may name multiple fields from the selected PE without repeating the search.

### 8.3 Grouped reduction

A grouped reduction partitions responders by one or more key expressions and reduces values within each group. Group order is ascending key order where the key type defines a total order. Each group's internal reduction order is ascending PE identity.

The output is a new associative space or an explicitly named destination. Destination construction is atomic with respect to failure.

### 8.4 Scans

Inclusive and exclusive scans require an explicit logical order and combining operation. Scans over an unordered responder set are invalid. A scan's output remains associated with the source PEs.

### 8.5 Compaction

Compaction produces a dense sequence of values from responders in an explicit logical order. It differs from masked selection, which preserves the original domain.

### 8.6 Indexed communication

Gather reads values through valid PE handles or indices. Scatter writes through handles or indices. Active invalid access is an error; inactive access is not evaluated. Ordinary scatter rejects duplicate destinations. Accumulating scatter requires a specified reduction and uses its ordering contract.

## 9. Arrays and structural values

An array has an element type, a finite shape, and one element at each valid zero-based index. A scalar is rank zero. A zero dimension produces an empty array.

Elementwise operands must have identical shapes, except for scalar extension. Array truth is never implicit. Indexing, slicing, gathering, reshape, and transpose produce values. Logical array order varies the last coordinate fastest. Negative indexing and implicit wrapping are not part of v0.1.

Array values are immutable. Mutable storage locations may hold arrays and may be updated under the snapshot rules of section 5. Slices and gathers do not create mutable views in v0.1.

Array reductions traverse reduced coordinates in logical order with the last reduced axis varying fastest. Listing axes in a different order does not change traversal.

## 10. Functions and modules

Functions declare scalar, array, field, mask, space, handle, and optional parameters explicitly. Parameters identify whether a function reads or mutates a space or storage location. Pure functions cannot access mutable state or external resources.

A function invoked in distributed context is evaluated independently at active PEs unless declared collective. A collective function receives one execution context and may search, reduce, or change the active set. Calling a collective function from element-local evaluation is invalid.

Overloading is resolved statically by parameter types and modes. Implicit numeric conversion does not participate. Recursion is permitted for scalar control; recursive collective calls must preserve and restore execution context at every call boundary.

Modules provide explicit imports and exports. Initialization order, package discovery, foreign-function interfaces, and a standard module registry are **Open**.

## 11. Dynamic spaces and relationships

Allocation and release are collective space operations. They are atomic and cannot expose partly initialized records. A released handle becomes invalid immediately. Reusing physical storage does not revive old handles.

A relationship between spaces is represented by typed PE handles or collections of handles. Deleting a target does not silently redirect a relationship. Dereferencing a stale relationship is an error unless the operation explicitly returns an optional result.

Changing occupancy invalidates no mask value as a value, but application of a saved mask automatically intersects it with current occupancy. Thus released PEs never re-enter execution through an old mask, while later allocations are not included in that old mask.

## 12. Errors and optional results

Rana distinguishes semantic errors from ordinary absence.

- Overflow, division by zero, incompatible domains or shapes, active invalid access, stale handles, and unresolved write conflicts are errors.
- An empty search, failed optional lookup, or empty `argmin` is ordinary absence.

An operation that fails commits no writes from that operation. Element-local errors arising at any active PE fail the collective operation. Inactive PEs cannot generate errors from suppressed computation.

The concrete error-propagation syntax is provisional. Implementations must at minimum provide fail-fast propagation with source location and operation context. Recoverable error values and per-PE error collection are **Open**.

## 13. Input and output

Core computation does not depend on external I/O. Loading data into a space must validate schema, types, capacity, and keys before committing any records. Output from a collective context requires an explicit ordering or is defined as unordered.

Individual-responder I/O is sequenced by that iteration's selection order. Concurrent external effects, transactions, files, networking, and serialization formats are outside the portable v0.1 core.

## 14. Introspection and diagnostics

Portable execution-state introspection includes:

```rana
active_mask()             // current active set
responder_mask()          // most recently established responders
responder_count()         // number of current responders
occupied_mask(Employee)   // current occupancy of a space
current_space()           // current associative-space identity
```

These operations observe state but do not change it. A returned mask is a value snapshot.

Implementations may provide optional diagnostics such as responder visualization, space inspection, execution-context stacks, trace events, and backend statistics. Diagnostic facilities must not be required for program correctness and may not change portable results.

## 15. Execution and memory model

The abstract layout of a space is field-addressable PE-local storage with identical schemas. Its physical representation may be array-of-structures, structure-of-arrays, tiled, compressed, distributed, or fused with computation.

An implementation may choose scalar, SIMD, multithreaded, GPU, distributed, or mixed execution. It may fuse operations, avoid mask materialization, tile spaces, or reorder independent work only when it preserves:

- responder and context semantics;
- snapshot updates and failure atomicity;
- defined reduction and selection order;
- occupancy and handle validity;
- suppressed evaluation at inactive PEs; and
- all permitted observable effects.

Host-device placement, residency, transfer, and synchronization are implementation concerns unless exposed through an optional nonportable module.

## 16. Conformance

A conforming implementation must:

1. implement every normative v0.1 rule it claims to support;
2. reject programs that depend on unsupported required features;
3. produce the same portable values and errors regardless of backend;
4. document resource limits without changing language semantics; and
5. identify extensions so they cannot be mistaken for portable v0.1 features.

The conformance suite should cover nested context restoration, empty responders, active-only errors, snapshot updates, duplicate writes, deterministic reductions, extrema ties, dynamic occupancy, stale handles, individual-responder iteration, grouped reduction, and introspection.

## 17. Illustrative example

The following program gives every qualifying employee a raise and reports the highest resulting salary. Syntax is provisional; behavior is normative according to the preceding rules.

```rana
space Employee(capacity: 100_000) {
    id: i64;
    department: i64;
    active: bool;
    salary: f64;
}

fn apply_raise(mut employees: Employee, dept: i64, rate: f64) -> optional<(i64, f64)> {
    where employees (active and department == dept) {
        salary = salary * (1.0 + rate);
    }

    let eligible = search employees where active and department == dept;
    let best = argmax(eligible, by: salary, tie: lowest_identity);

    if best is none {
        return none;
    }
    return some((best.handle.id, best.value));
}
```

The update is collective and atomic. Only responders evaluate the multiplication. The second search is independent of the restored context from `where`. `argmax` returns none for no responders and deterministically selects the lowest-identity PE on a tie.

## Appendix A. v0.1 feature boundary

The following are part of the v0.1 semantic core:

- associative spaces and logical PEs;
- occupancy, active sets, responder sets, and nested restoration;
- first-class domain-bound masks;
- associative search and masked execution;
- collective and individual-responder control flow;
- deterministic reductions, associated extrema, grouping, and indexed communication;
- checked core numerics;
- snapshot updates and atomic operation failure;
- dynamic membership and generation-checked handles; and
- core execution-state introspection.

The following are deferred extensions:

- multiple concurrent instruction streams;
- distributed-memory consistency and failure recovery;
- relaxed or nondeterministic numerical modes;
- sparse array and sparse-space storage contracts;
- automatic differentiation;
- user-defined effect systems and concurrency;
- hardware-specific vector widths and accelerator intrinsics; and
- standardized package, FFI, and external I/O systems.

## Appendix B. Historical ASC mapping

| ASC concept | Rana v0.1 concept |
|---|---|
| `PARALLEL` variable | distributed space field |
| `ASSOCIATE` | associative-space schema |
| `SETSCOPE` | nested active context / `where` |
| parallel `IF` | associative conditional |
| `ANY` / `IF-NOT-ANY` | explicit responder tests and empty branch |
| `FOR`, `WHILE` | collective associative loops |
| `NEXT` | individual-responder iteration |
| `GET` | selected-handle retrieval |
| `COUNT` | responder count reduction |
| `MINVAL`, `MAXVAL` | scalar extrema reductions |
| `MINDEX`, `MAXDEX` | deterministic `argmin`, `argmax` |
| `ALLOCATE`, `RELEASE` | dynamic space occupancy |

This mapping preserves computational meaning without requiring compatibility with historical ASC syntax.
