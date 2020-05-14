# Zig-Cassowary

A port of the Cassowary constraint solver "Kiwi" to zig.

The two main differences from Kiwi are that shared data pointers are not used
and zig doesn't support operator overloading or the allocator "magic" of C++.

In practice this means constraints and expressions must manually be allocated
and deallocated and pointers to variables used instead of copying them.


### Example

```zig
const std = @import("std");
const kiwi = @import("kiwi.zig");

var buf: [10000]u8 = undefined;
const allocator = &std.heap.FixedBufferAllocator.init(&buf).allocator;
var solver = kiwi.Solver.init(allocator);
defer solver.deinit();

var width = kiwi.Variable{.name="width"};
var height = kiwi.Variable{.name="height"};

try solver.addVariable(&width, kiwi.Strength.medium);
try solver.addVariable(&height, kiwi.Strength.medium);

// 16 * width = 9 * height
var aspect = try solver.buildConstraint(
    width.mul(16), .eq, height.mul(9), kiwi.Strength.weak);
defer aspect.deinit();
try solver.addConstraint(&aspect);

try solver.suggestValue(width, 240);
solver.updateVariables(); // Updates the interal value of every variable

std.testing.expect(height.value == 135);

```

### License

Licensed under the same BSD license as the original Kiwi. Since it is a direct
port the original the Kiwi license may be obtained from https://github.com/nucleic/kiwi.


### Donate

If you would like to say thanks, please send a donation via Stripe or Paypal
at https://codelv.com/donate/ or email me if you want an invoice.

Thank you!
