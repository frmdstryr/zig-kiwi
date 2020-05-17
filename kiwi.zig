// -------------------------------------------------------------------------- //
// Copyright (c) 2020, Jairus Martin jrm@codelv.com                           //
// Ported from the C++ implementation named Kiwi                              //
// Copyright (c) 2013-2017, Nucleic Development Team.                         //
// Distributed under the terms of the BSD License.                            //
// -------------------------------------------------------------------------- //
const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

pub const tolerance = 1.0e-8;


pub const Variable = struct {
    name: []const u8 = "",
    value: f32 = 0.0,
    context: ?usize = null,

    // -----------------------------------------------------------------------
    // Multiply, divide, and invert create terms
    // -----------------------------------------------------------------------
    pub inline fn mul(self: *Variable, coefficient: f32) Term {
        return Term{.variable=self, .coefficient=coefficient};
    }

    pub inline fn div(self: *Variable, denominator: f32) Term {
        return self.mul(1.0/denominator);
    }

    // Unary invert
    pub inline fn invert(self: *Variable) Term {
        return self.mul(-1.0);
    }

    // -----------------------------------------------------------------------
    // Add and subtract create expressions
    // -----------------------------------------------------------------------
    pub inline fn add(self: *Variable, allocator: *Allocator, other: var) !Expression {
        var term = Term{.variable=self};
        return term.add(allocator, other);
    }

    pub inline fn sub(self: *Variable, allocator: *Allocator, other: var) !Expression {
        var term = Term{.variable=self};
        return term.sub(allocator, other);
    }

    // -----------------------------------------------------------------------
    // Constraint API
    // -----------------------------------------------------------------------
//     pub fn buildConstraint(self: Variable, allocator: *Allocator,
//                            op: Constraint.Op, value: var,
//                            strength: f32) !Constraint {
//         return Constraint{
//             .expression=try Expression.init(allocator, &[_]Term{
//                 Term.init(self), Term.init(value)}),
//             .op = op,
//             .strength = strength,
//         };
//     }
//
//     pub fn eql(self: Variable, allocator: *Allocator,
//                value: var, strength: f32) !Constraint {
//         return self.buildConstraint(allocator, .eq, value, strength);
//     }
//
//     pub fn lte(self: Variable, allocator: *Allocator,
//               value: var, strength: f32) !Constraint {
//         return self.buildConstraint(allocator, .lte, value, strength);
//     }
//
//     pub fn gte(self: Variable, allocator: *Allocator,
//               value: var, strength: f32) !Constraint {
//         return self.buildConstraint(allocator, .gte, value, strength);
//     }

};

pub const Term = struct {
    coefficient: f32 = 1.0,
    variable: *Variable,

    pub inline fn value(self: *const Term) f32 {
        return self.coefficient * self.variable.value;
    }

    // -----------------------------------------------------------------------
    // Multiply, divide, and invert create terms
    // -----------------------------------------------------------------------
    pub inline fn mul(self: *Term, coefficient: f32) Term {
        return Term{
            .variable=self.variable,
            .coefficient=self.coefficient * coefficient
        };
    }

    pub inline fn div(self: *Term, denominator: f32) Term {
        return self.mul(1.0/denominator);
    }

    // Unary invert
    pub inline fn invert(self: *Term) Term {
        return self.mul(-1.0);
    }

    // -----------------------------------------------------------------------
    // Add and subtract create expressions
    // -----------------------------------------------------------------------
    pub inline fn add(self: *Term, allocator: *Allocator, other: var) !Expression {
        // No need to allocate here since expr.add makes a copy
        var terms = Expression.Terms.fromOwnedSlice(allocator, &[_]Term{self.*});
        var expr = Expression{.terms=terms};
        return expr.add(allocator, other);
    }

    pub inline fn sub(self: *Term, allocator: *Allocator, other: var) !Expression {
        // No need to allocate here since expr.sub makes a copy
        var terms = Expression.Terms.fromOwnedSlice(allocator, &[_]Term{self.*});
        var expr = Expression{.terms=terms};
        return expr.sub(allocator, other);
    }

    // -----------------------------------------------------------------------
    // Constraint API
    // -----------------------------------------------------------------------
//     pub fn buildConstraint(self: Term, allocator: *Allocator, op: Constraint.Op,
//                            value: var, strength: f32) !Constraint {
//         return Constraint{
//             .expression=try Expression.init(allocator,
//                 &[_]Term{self, Term.init(value)}),
//             .op = op,
//             .strength = strength,
//         };
//     }
//
//     pub inline fn eql(self: Term, allocator: *Allocator,
//                value: var, strength: f32) !Constraint {
//         return self.buildConstraint(allocator, .eq, value, strength);
//     }
//
//     pub inline fn lte(self: Term, allocator: *Allocator,
//                value: var, strength: f32) !Constraint {
//         return self.buildConstraint(allocator, .lte, value, strength);
//     }
//
//     pub inline fn gte(self: Term, allocator: *Allocator,
//                value: var, strength: f32) !Constraint {
//         return self.buildConstraint(allocator, .gte, value, strength);
//     }
};


pub const Expression = struct {
    pub const Vars = std.AutoHashMap(*Variable, f32);
    pub const Terms = std.ArrayList(Term);
    terms: Terms,
    constant: f32 = 0.0,

    // Create a reduced expression from a slice of terms
    pub fn init(allocator: *Allocator, args: []Term) !Expression {
        return initConstant(allocator, args, 0.0);
    }

    // Create a reduced expression from a slice of terms with a constant
    pub fn initConstant(allocator: *Allocator, args: []Term, constant: f32) !Expression {
        var vars = Vars.init(allocator);
        defer vars.deinit();
        for (args) |term| {
            var entry = try vars.getOrPutValue(term.variable, 0.0);
            entry.value += term.coefficient;
        }
        var terms = try Terms.initCapacity(allocator, vars.size);
        var it = vars.iterator();
        while (it.next()) |entry| {
            terms.appendAssumeCapacity(Term{
                .variable=entry.key,
                .coefficient=entry.value
            });
        }
        return Expression{
            .terms = terms,
            .constant = constant,
        };
    }

    // Create a reduced expression by from the terms of an existing expression
    // For example 3x + 2x + y will be reduced to two terms 5x + y
    pub fn reduce(expr: *Expression, allocator: *Allocator) !Expression {
        var vars = Vars.init(allocator);
        defer vars.deinit();
        for (expr.terms.items) |term| {
            var entry = try vars.getOrPutValue(term.variable, 0.0);
            entry.value += term.coefficient;
        }
        var terms = try Terms.initCapacity(allocator, vars.size);
        var it = vars.iterator();
        while (it.next()) |entry| {
            terms.appendAssumeCapacity(Term{
                .variable=entry.key,
                .coefficient=entry.value
            });
        }
        return Expression{
            .terms = terms,
            .constant = expr.constant,
        };
    }

    pub inline fn deinit(self: *Expression) void {
        self.terms.deinit();
    }

    // Evaluate the expression using the variable's values
    pub fn value(self: *const Expression) f32 {
        var result = self.constant;
        for (self.terms.items) |term| {
            result += term.value();
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Multiply, divide, and invert
    // -----------------------------------------------------------------------
    pub fn mul(self: *Expression, allocator: *Allocator, coefficient: f32) !Expression {
        var terms = try Terms.initCapacity(allocator, self.terms.items.len);
        for (self.terms.items) |*term| {
            terms.appendAssumeCapacity(term.mul(coefficient));
        }
        return Expression{
            .terms=terms,
            .constant=self.constant * coefficient,
        };
    }

    pub inline fn div(self: *Expression, allocator: *Allocator, denominator: f32) !Expression {
        return self.mul(allocator, 1.0/denominator);
    }

    // Unary invert
    pub inline fn invert(self: *Expression, allocator: *Allocator) !Expression {
        return self.mul(allocator, -1.0);
    }

    // -----------------------------------------------------------------------
    // Add and subtract
    // -----------------------------------------------------------------------
    pub fn addAndMul(self: *Expression, allocator: *Allocator,
                     other: var, coefficient: f32) !Expression {
        comptime const T = @TypeOf(other);

        const size = self.terms.items.len + switch(T) {
            Expression => other.terms.items.len,
            *Term, *Variable => 1,
            Term, Variable => @compileError("Pass the Term/Variable as a reference"),
            else => 0,
        };

        // TODO: Optimize this by avoiding creating two sets of terms
        var terms = try Terms.initCapacity(allocator, size);
        defer terms.deinit();
        for (self.terms.items) |term| {
            terms.appendAssumeCapacity(term);
        }
        var constant: f32 = self.constant;
        switch (T) {
            Expression => {
                for (other.terms.items) |*term| {
                    terms.appendAssumeCapacity(term.mul(coefficient));
                }
                constant += other.constant * coefficient;
            },
            *Term, *Variable => {
                terms.appendAssumeCapacity(other.mul(coefficient));
            },
            else => {
                constant += @as(f32, other) * coefficient;
            }
        }
        return Expression.initConstant(allocator, terms.items, constant);
    }

    pub fn add(self: *Expression, allocator: *Allocator, other: var) !Expression {
        return self.addAndMul(allocator, other, 1);
    }

    pub fn sub(self: *Expression, allocator: *Allocator, other: var) !Expression {
        return self.addAndMul(allocator, other, -1);
    }

};

pub const Strength = struct {
    pub inline fn create(a: f32, b: f32, c: f32) f32 {
        return createWeighted(a, b, c, 1.0);
    }

    pub inline fn createWeighted(a: f32, b: f32, c: f32, w: f32) f32 {
        comptime const low = @as(f32, 0.0);
        comptime const high = @as(f32, 1000.0);
        var r: f32 = 0.0;

        r += std.math.clamp(a * w, low, high) * 1000000.0;
        r += std.math.clamp(b * w, low, high) * 1000.0;
        r += std.math.clamp(c * w, low, high);
        return r;
    }

    pub const required = create(1000.0, 1000.0, 1000.0);
    pub const strong   = create(1.0, 0.0, 0.0);
    pub const medium   = create(0.0, 1.0, 0.0);
    pub const weak     = create(0.0, 0.0, 1.0);

    pub inline fn clamp(value: f32) f32 {
        return std.math.clamp(value, 0.0, required);
    }
};

pub inline fn isNearZero(value: f32) bool {
    return std.math.approxEq(f32, value, 0.0, tolerance);
}

pub const Constraint = struct {
    pub const Op = enum {
        eq,
        gte,
        lte,
        pub fn str(op: Op) []const u8 {
            return switch (op) {
                .eq => "==",
                .gte => ">=",
                .lte => "<=",
            };
        }
    };
    op: Op,
    strength:  f32 = 0.0,
    expression: Expression,

    pub inline fn deinit(self: *Constraint) void {
        self.expression.deinit();
    }

     pub fn dumps(self: *Constraint) void {
        const expr = &self.expression;
        const last = expr.terms.items.len - 1;
        for (expr.terms.items) |term, i| {
            if (isNearZero(term.coefficient - 1.0)) {
                std.debug.warn("{}", .{term.variable.name});
            } else {
                std.debug.warn("{d} * {}", .{term.coefficient, term.variable.name});
            }
            if (i != last) {
                 std.debug.warn(" + ", .{});
            }
        }
        if (!isNearZero(expr.constant)) {
            std.debug.warn(" + {d}", .{expr.constant});
        }
        std.debug.warn(" {} 0 | strength = {d}", .{self.op.str(), self.strength});
        std.debug.warn("\n", .{});
    }

};

const Symbol = struct {
    pub const Type = enum {
        Invalid,
        External,
        Slack,
        Error,
        Dummy,

        pub fn str(self: Type) []const u8 {
            return switch(self) {
                .Invalid => "i",
                .External => "v",
                .Slack => "s",
                .Error => "e",
                .Dummy => "d",
            };
        }
    };

    id: u32 = 0,
    tp: Type = .Invalid,

    pub inline fn hash(self: Symbol) u32 {
        return self.id;
    }

    pub inline fn lte(self: Symbol, other: Symbol) bool {
        return self.id < other.id;
    }

    pub inline fn eql(self: Symbol, other: Symbol) bool {
        return self.id == other.id;
    }

    pub const Invalid = Symbol{.id=0, .tp=.Invalid};

    pub fn dumps(self: Symbol) void {
        std.debug.warn("{}{}", .{self.tp.str(), self.id});
    }

};


const Row = struct {
    pub const CellMap = std.HashMap(Symbol, f32, Symbol.hash, Symbol.eql);

    cells: CellMap,
    constant: f32 = 0.0,

    pub fn init(allocator: *Allocator) Row {
        return Row{
            .cells = CellMap.init(allocator),
        };
    }

    pub fn deinit(self: *Row) void {
        self.cells.deinit();
    }

    // Create a clone of the row using the original allocator
    pub fn clone(self: Row) !Row {
        return Row{
            .cells = try self.cells.clone(),
            .constant = self.constant,
        };
    }

    // Add a constant value to the row constant.
    // The new value of the constant is returned.
    pub inline fn add(self: *Row, value: f32) f32 {
        self.constant += value;
        return self.constant;
    }

    // Insert a symbol into the row with a given coefficient.
    //
    // If the symbol already exists in the row, the coefficient will be
    // added to the existing coefficient. If the resulting coefficient
    // is zero, the symbol will be removed from the row.
    pub inline fn insertSymbol(self: *Row, symbol: Symbol, coefficient: f32) !void {
        const entry = try self.cells.getOrPutValue(symbol, 0.0);
        entry.value += coefficient;
        if (isNearZero(entry.value)) {
            self.removeSymbol(symbol);
        }
    }

    // Remove the given symbol from the row.
    pub inline fn removeSymbol(self: *Row, symbol: Symbol) void {
        _ = self.cells.remove(symbol);
    }

    // Insert a row into this row with a given coefficient.
    //
    // The constant and the cells of the other row will be multiplied by
    // the coefficient and added to this row. Any cell with a resulting
    // coefficient of zero will be removed from the row.
    pub fn insertRow(self: *Row, row: *const Row, coefficient: f32) !void {
        self.constant += row.constant * coefficient;
        var it = row.cells.iterator();
        while (it.next()) |item| {
            try self.insertSymbol(item.key, item.value * coefficient);
        }
    }

    // Reverse the sign of the constant and all cells in the row.
    pub fn reverseSign(self: *Row) void {
        self.constant = -self.constant;
        var it = self.cells.iterator();
        while (it.next()) |entry| {
            entry.value = -entry.value;
        }
    }

    // Solve the row for the given symbol.

    // This method assumes the row is of the form a * x + b * y + c = 0
    // and (assuming solve for x) will modify the row to represent the
    // right hand side of x = -b/a * y - c / a. The target symbol will
    // be removed from the row, and the constant and other cells will
    // be multiplied by the negative inverse of the target coefficient.
    //
    // The given symbol *must* exist in the row.
    pub fn solveForSymbol(self: *Row, symbol: Symbol) !void {
        if (self.cells.get(symbol)) |cell| {
            self.removeSymbol(symbol);
            const coeff = -1.0 / cell.value;
            self.constant *= coeff;
            var it = self.cells.iterator();
            while (it.next()) |entry| {
                entry.value *= coeff;
            }
        } else {
            return error.CannotSolveForUnknownSymbol;
        }
    }

    // Solve the row for the given symbols.
    //
    // This method assumes the row is of the form x = b * y + c and will
    // solve the row such that y = x / b - c / b. The rhs symbol will be
    // removed from the row, the lhs added, and the result divided by the
    // negative inverse of the rhs coefficient.
    //
    // The lhs symbol *must not* exist in the row, and the rhs symbol
    // *must* exist in the row.
    pub fn solveFor(self: *Row, lhs: Symbol, rhs: Symbol) !void {
        assert(!self.cells.contains(lhs));
        try self.insertSymbol(lhs, -1.0);
        return self.solveForSymbol(rhs);
    }


    // Get the coefficient for the given symbol.
    // If the symbol does not exist in the row, zero will be returned.
    pub inline fn coefficientFor(self: Row, symbol: Symbol) f32 {
        return if (self.cells.getValue(symbol)) |c| c else 0.0;
    }

    // Substitute a symbol with the data from another row.
    //
    // Given a row of the form a * x + b and a substitution of the
    // form x = 3 * y + c the row will be updated to reflect the
    // expression 3 * a * y + a * c + b.
    //
    // If the symbol does not exist in the row, this is a no-op.
    pub fn substitute(self: *Row, symbol: Symbol, row: *const Row) !void {
        if (self.cells.get(symbol)) |cell| {
            self.removeSymbol(symbol);
            try self.insertRow(row, cell.value);
        }
    }

    // Test whether a row is composed of all dummy variables.
    pub fn isAllDummies(self: Row) bool {
        var it = self.cells.iterator();
        while (it.next()) |entry| {
            if (entry.key.tp != .Dummy) return false;
        }
        return true;
    }

    pub fn dumps(self: Row) void {
        var it = self.cells.iterator();
        std.debug.warn("{d}", .{self.constant});
        while (it.next()) |entry| {
            if (isNearZero(entry.value - 1.0)) {
                std.debug.warn(" + ", .{});
            } else {
                std.debug.warn(" + {d} * ", .{entry.value});
            }
            entry.key.dumps();
        }
        std.debug.warn("\n", .{});
    }

};


pub const Solver = struct {

    pub const Tag = struct {
        marker: Symbol = Symbol.Invalid,
        other: Symbol = Symbol.Invalid,
    };

    pub const EditInfo = struct {
        tag: *Tag,
        constraint: *Constraint,
        constant: f32 = 0.0,
    };

    pub const VarMap = std.AutoHashMap(*Variable, Symbol);
    pub const RowMap = std.HashMap(Symbol, Row, Symbol.hash, Symbol.eql);
    pub const SymbolList = std.ArrayList(Symbol);
    pub const ConstraintMap = std.AutoHashMap(*Constraint, Tag);
    pub const EditMap = std.AutoHashMap(*Variable, EditInfo);

    // Fields
    allocator: *Allocator,
    cns: ConstraintMap,
    rows: RowMap,
    vars: VarMap,
    edits: EditMap,
    infeasible_rows: SymbolList,
    objective: Row,
    artificial: ?*Row = null,
    _next_id: u32 = 1,

    pub fn init(allocator: *Allocator) Solver {
        return Solver{
            .allocator = allocator,
            .cns = ConstraintMap.init(allocator),
            .rows = RowMap.init(allocator),
            .vars = VarMap.init(allocator),
            .edits = EditMap.init(allocator),
            .infeasible_rows = SymbolList.init(allocator),
            .objective = Row.init(allocator),
        };
    }

    pub fn deinit(self: *Solver) void {
        self.clearRows();
    }

    // -----------------------------------------------------------------------
    // Builder API
    // -----------------------------------------------------------------------
    pub fn buildConstraint(self: *Solver, lhs: var, op: Constraint.Op,
                                  rhs: var, strength: f32) !Constraint {
        var expr = switch (@TypeOf(lhs)) {
            *Expression, *Variable, *Term => try lhs.sub(self.allocator, rhs),
            Expression, Variable, Term => @compileError("Pass a reference"),
            else => @compileError("Invalid lhs expression"),
        };
        return Constraint{
            .expression = expr,
            .op = op,
            .strength = strength,
        };
    }

    // -----------------------------------------------------------------------
    // Solver API
    // -----------------------------------------------------------------------

    // Add a constraint to the solver.
    // Throws
    // ------
    // DuplicateConstraint
    //     The given constraint has already been added to the solver.
    //
    // UnsatisfiableConstraint
    //    The given constraint is required and cannot be satisfied.
    pub fn addConstraint(self: *Solver, constraint: *Constraint) !void {
        if (self.cns.contains(constraint)) {
            return error.DuplicateConstraint;
        }
        // Creating a row causes symbols to be reserved for the variables
        // in the constraint. If this method exits with an exception,
        // then its possible those variables will linger in the var map.
        // Since its likely that those variables will be used in other
        // constraints and since exceptional conditions are uncommon,
        // i'm not too worried about aggressive cleanup of the var map.
        var tag = Tag{};
        var row = try self.createRow(constraint, &tag);
        var subject = self.chooseSubject(row, &tag);

        // If chooseSubject could not find a valid entering symbol, one
        // last option is available if the entire row is composed of
        // dummy variables. If the constant of the row is zero, then
        // this represents redundant constraints and the new dummy
        // marker can enter the basis. If the constant is non-zero,
        // then it represents an unsatisfiable constraint.
        if (subject.tp == .Invalid and row.isAllDummies()) {
            if (!isNearZero(row.constant)) {
                return error.UnsatisfiableConstraint;
            }
            subject = tag.marker;
        }

        // If an entering symbol still isn't found, then the row must
        // be added using an artificial variable. If that fails, then
        // the row represents an unsatisfiable constraint.
        if (subject.tp == .Invalid) {
            if (!try self.addWithArtificialVariable(row)) {
                return error.UnsatisfiableConstraint;
            }
        } else {
            try row.solveForSymbol(subject);
            try self.substitute(subject, &row);
            _ = try self.rows.put(subject, row);
        }

        _ = try self.cns.put(constraint, tag);

        // Optimizing after each constraint is added performs less
        // aggregate work due to a smaller average system size. It
        // also ensures the solver remains in a consistent state.
        try self.optimize(&self.objective);
    }


    // Remove a constraint from the solver.
    pub fn removeConstraint(self: *Solver, constraint: *Constraint) !void {
        if (self.cns.remove(constraint)) |e| {
            const tag = e.value;

            // Remove the error effects from the objective function
            // *before* pivoting, or substitutions into the objective
            // will lead to incorrect solver results.
            try self.removeConstraintEffects(constraint, tag);

            // If the marker is basic, simply drop the row. Otherwise,
            // pivot the marker into the basis and then drop the row.
            if (self.rows.remove(tag.marker)) |_| {
                // Already removed
            } else if (self.getMarkerLeavingRow(tag.marker)) |entry| {
                _ = self.rows.remove(entry.key);
                const row = &entry.value;
                try row.solveFor(entry.key, tag.marker);
                try self.substitute(tag.marker, row);
            } else {
                return error.InternalSolverError; // Unable to find leaving row
            }

            // Optimizing after each constraint is removed ensures that the
            // solver remains consistent. It makes the solver api easier to
            // use at a small tradeoff for speed.
            try self.optimize(&self.objective);
        } else {
            return error.UnknownConstraint;
        }
    }

    // Test whether a constraint has been added to the solver.
    pub fn hasConstraint(self: *Solver, constraint: *Constraint) bool {
        return self.cns.contains(constraint);
    }

    // Add an edit variable to the solver.
    //
    // This method should be called before the `suggestValue` method is
    // used to supply a suggested value for the given edit variable.
    //
    // Throws
    // ------
    // - DuplicateVariable
    //     The given edit variable has already been added to the solver.
    //
    // - BadRequiredStrength
    //     The given strength is >= required.
    pub fn addVariable(self: *Solver, variable: *Variable, strength: f32) !void {
        if (self.edits.contains(variable)) {
            return error.DuplicateVariable;
        }
        const s = Strength.clamp(strength);
        if (s >= Strength.required) {
            return error.BadRequiredStrength;
        }
        const constraint = try self.allocator.create(Constraint);
        errdefer self.allocator.destroy(constraint);

        var terms = try Expression.Terms.initCapacity(self.allocator, 1);
        terms.appendAssumeCapacity(Term{.variable=variable});
        errdefer terms.deinit();

        constraint.* = Constraint{
            .expression = Expression{.terms=terms},
            .op = .eq,
            .strength = strength,
        };

        try self.addConstraint(constraint);
        var info = EditInfo{
            .tag = &self.cns.get(constraint).?.value,
            .constraint = constraint,
        };
        _ = try self.edits.put(variable, info);
    }

    // Remove an edit variable from the solver.
    // Throws
    // ------
    // - UnknownVariable
    //   The given edit variable has not been added to the solver.
    pub fn removeVariable(self: *Solver, variable: *Variable) !void {
        if (self.edits.remove(variable)) |entry| {
            const constraint = entry.value.constraint;
            try self.removeConstraint(constraint);
            constraint.deinit();
            self.allocator.destroy(constraint);
        } else {
            return error.UnknownVariable;
        }
    }

    // Test whether an edit variable has been added to the solver.
    pub fn hasVariable(self: *Solver, variable: *Variable) bool {
        return self.edits.contains(variable);
    }

    // Suggest a value for the given edit variable.
    //
    // This method should be used after an edit variable as been added to
    // the solver in order to suggest the value for that variable. After
    // all suggestions have been made, the `solve` method can be used to
    // update the values of all variables.
    //
    // Throws
    // ------
    // UnknownVariable
    //    The given edit variable has not been added to the solver.
    //
    pub fn suggestValue(self: *Solver, variable: *Variable, value: f32) !void {
        if (self.edits.getValue(variable)) |*info| {
            const delta = value - info.constant;
            info.constant = value;

            // TODO: Ensure that dualOptimize is always called
            try self.updateSuggestion(info, delta);
            try self.dualOptimize();
        } else {
            return error.UnknownVariable;
        }
    }

    // Helper function for suggested value
    fn updateSuggestion(self: *Solver, info: *const EditInfo, delta: f32) !void {
        // Check first if the positive error variable is basic.
        if (self.rows.get(info.tag.marker)) |entry| {
            if (entry.value.add(-delta) < 0.0) {
                try self.infeasible_rows.append(entry.key);
            }
            return;
        }

        // Check next if the negative error variable is basic.
        if (self.rows.get(info.tag.other)) |entry| {
            if (entry.value.add(delta) < 0.0) {
                try self.infeasible_rows.append(entry.key);
            }
            return;
        }

        // Otherwise update each row where the error variables exist.
        var it = self.rows.iterator();
        while (it.next()) |entry| {
            const coeff = entry.value.coefficientFor(info.tag.marker);
            if (coeff != 0.0
                    and entry.value.add(delta * coeff) < 0.0
                    and entry.key.tp != .External) {
                try self.infeasible_rows.append(entry.key);
            }
        }
    }

    // Update the values of the external solver variables.
    pub fn updateVariables(self: *Solver) void {
        var it = self.vars.iterator();
        while (it.next()) |entry| {
            const v = entry.key;
            const sym = entry.value;
            v.value = if (self.rows.getValue(sym)) |row| row.constant else 0.0;
        }
    }

    // Reset the solver to the empty starting condition.
    // This method resets the internal solver state to the empty starting
    // condition, as if no constraints or edit variables have been added.
    // This can be faster than deleting the solver and creating a new one
    // when the entire system must change, since it can avoid unecessary
    // heap (de)allocations.
    pub fn reset(self: *Solver) void {
        self.clearRows();
        self.cns.clear();
        self.vars.clear();
        self.edits.clear();
        self.infeasible_rows.deinit();
        self.objective.deinit();
        self.objective = Row.init(self.allocator);
        self.artificial = null;
        self._next_id = 1;
    }

    // -----------------------------------------------------------------------
    // Internal API
    // -----------------------------------------------------------------------
    fn clearRows(self: *Solver) void {
        var it = self.rows.iterator();
        while (it.next()) |entry| {
            entry.value.deinit();
        }
        self.rows.clear();
    }

    // Get the symbol for the given variable.
    // If a symbol does not exist for the variable, one will be created.
    fn getVarSymbol(self: *Solver, variable: *Variable) !Symbol {
        if (self.vars.getValue(variable)) |symbol| {
            return symbol;
        }
        const symbol = self.createSymbol(.External);
        _ = try self.vars.put(variable, symbol);
        return symbol;
    }

    // Create a new symbol using a generated id
    inline fn createSymbol(self: *Solver, tp: Symbol.Type) Symbol {
        const id = self._next_id;
        self._next_id += 1;
        return Symbol{.id = id, .tp = tp};
    }

    // Create a new Row object for the given constraint.
    //
    // The terms in the constraint will be converted to cells in the row.
    // Any term in the constraint with a coefficient of zero is ignored.
    // This method uses the `getVarSymbol` method to get the symbol for
    // the variables added to the row. If the symbol for a given cell
    // variable is basic, the cell variable will be substituted with the
    // basic row.
    //
    // The necessary slack and error variables will be added to the row.
    // If the constant for the row is negative, the sign for the row
    // will be inverted so the constant becomes positive.
    //
    // The tag will be updated with the marker and error symbols to use
    // for tracking the movement of the constraint in the tableau.
    fn createRow(self: *Solver, constraint: *Constraint, tag: *Tag) !Row {
        const expr = &constraint.expression;
        const objective = &self.objective;
        var row = Row.init(self.allocator);
        row.constant = expr.constant;

        // Substitute the current basic variables into the row.
        for (expr.terms.items) |term| {
            if (isNearZero(term.coefficient)) continue;
            const symbol = try self.getVarSymbol(term.variable);
            if (self.rows.getValue(symbol)) |*r| {
                try row.insertRow(r, term.coefficient);
            } else {
                try row.insertSymbol(symbol, term.coefficient);
            }
        }

        // Add the necessary slack, error, and dummy variables.
        switch (constraint.op) {
            .lte, .gte => {
                const coeff: f32 = if (constraint.op == .lte) 1.0 else -1.0;
                const slack = self.createSymbol(.Slack);
                tag.marker = slack;
                try row.insertSymbol(slack, coeff);

                if (constraint.strength < Strength.required) {
                    const err = self.createSymbol(.Error);
                    tag.other = err;
                    try row.insertSymbol(err, -coeff);
                    try objective.insertSymbol(err, constraint.strength);
                }
            },
            .eq => {
                if (constraint.strength < Strength.required) {
                    const errplus = self.createSymbol(.Error);
                    const errminus = self.createSymbol(.Error);
                    tag.marker = errplus;
                    tag.other = errminus;
                    try row.insertSymbol(errplus, -1.0); // v = eplus - eminus
                    try row.insertSymbol(errminus, 1.0); // v - eplus + eminus = 0
                    try objective.insertSymbol(errplus, constraint.strength);
                    try objective.insertSymbol(errminus, constraint.strength);
                } else {
                    const dummy = self.createSymbol(.Dummy);
                    tag.marker = dummy;
                    try row.insertSymbol(dummy, 1.0);
                }
            }
        }

        // Ensure the row as a positive constant.
        if (row.constant < 0.0) {
            row.reverseSign();
        }

        return row;
    }

    // Choose the subject for solving for the row.
    //
    // This method will choose the best subject for using as the solve
    // target for the row. An invalid symbol will be returned if there
    // is no valid target.
    //
    // The symbols are chosen according to the following precedence:
    //
    // 1) The first symbol representing an external variable.
    // 2) A negative slack or error tag variable.
    //
    // If a subject cannot be found, no symbol is returned.
    fn chooseSubject(self: *Solver, row: Row, tag: *const Tag) Symbol {
        var it = row.cells.iterator();
        while (it.next()) |entry| {
            if (entry.key.tp == .External) {
                return entry.key;
            }
        }
        if ((tag.marker.tp == .Slack or tag.marker.tp == .Error)
                and row.coefficientFor(tag.marker) < 0.0) {
            return tag.marker;
        }

        if ((tag.other.tp == .Slack or tag.other.tp == .Error)
                and row.coefficientFor(tag.other) < 0.0) {
            return tag.other;
        }
        return Symbol{};

    }

    // Add the row to the tableau using an artificial variable.
    // This will return false if the constraint cannot be satisfied.
    fn addWithArtificialVariable(self: *Solver, row: Row) !bool {
        // Create and add the artificial variable to the tableau
        const art = self.createSymbol(.Slack);
        var artificial = try row.clone();
        _ = try self.rows.put(art, artificial);
        self.artificial = &artificial;

        // Optimize the artificial objective. This is successful
        // only if the artificial objective is optimized to zero.
        try self.optimize(&artificial);
        const success = isNearZero(artificial.constant);
        self.artificial = null;

        // If the artificial variable is not basic, pivot the row so that
        // it becomes basic. If the row is constant, exit early.
        if (self.rows.get(art)) |entry| {
            const r = &entry.value;
            _ = self.rows.remove(art);
            if (r.cells.size == 0) return success;
            if (self.getAnyPivotableSymbol(r)) |entering| {
                try r.solveFor(art, entering);
                try self.substitute(entering, r);
                _ = try self.rows.put(entering, entry.value);
            } else {
                return false; // unsatisfiable (will this ever happen?)
            }
        }

        // Remove the artificial variable from the tableau.
        var it = self.rows.iterator();
        while (it.next()) |entry| {
            entry.value.removeSymbol(art);
        }
        self.objective.removeSymbol(art);

        return success;
    }


    // Substitute the parametric symbol with the given row.
    //
    // This method will substitute all instances of the parametric symbol
    // in the tableau and the objective function with the given row.
    fn substitute(self: *Solver, symbol: Symbol, row: *const Row) !void {
        var it = self.rows.iterator();
        while (it.next()) |entry| {
            try entry.value.substitute(symbol, row);
            if (entry.key.tp != .External and entry.value.constant < 0.0) {
                try self.infeasible_rows.append(entry.key);
            }
        }

        try self.objective.substitute(symbol, row);
        if (self.artificial) |r| {
            try r.substitute(symbol, row);
        }
    }

    // Optimize the system for the given objective function.
    //
    // This method performs iterations of Phase 2 of the simplex method
    // until the objective function reaches a minimum.
    //
    // Throws
    // ------
    // InternalSolverError
    //    The value of the objective function is unbounded.
    fn optimize(self: *Solver, objective: *Row) !void {
        while (self.getEnteringSymbol(objective)) |entering| {
            if (self.getLeavingRow(entering)) |entry| {
                // pivot the entering symbol into the basis
                const leaving = entry.key;
                const row = &entry.value;
                assert(self.rows.remove(leaving) != null);
                try row.solveFor(leaving, entering);
                try self.substitute(entering, row);
                _ = try self.rows.put(entering, entry.value);
            } else {
                return error.InternalSolverError; // The objective is unbounded
            }
        }
    }

    // Optimize the system using the dual of the simplex method.
    //
    // The current state of the system should be such that the objective
    // function is optimal, but not feasible. This method will perform
    // an iteration of the dual simplex method to make the solution both
    // optimal and feasible.
    //
    // Throws
    // ------
    // InternalSolverError
    //    The system cannot be dual optimized.
    fn dualOptimize(self: *Solver) !void {
        while (self.infeasible_rows.popOrNull()) |leaving| {
            if (self.rows.get(leaving)) |entry| {
                const row = &entry.value;
                if (!isNearZero(row.constant) and row.constant < 0.0) {
                    if (self.getDualEnteringSymbol(row)) |entering| {
                        _ = self.rows.remove(leaving);
                        try row.solveFor(leaving, entering);
                        try self.substitute(entering, row);
                        _ = try self.rows.put(entering, entry.value);
                    } else {
                        return error.InternalSolverError; // Dual optimize fail
                    }
                }
            }
        }
    }

    // Compute the entering variable for a pivot operation.
    //
    // This method will return first symbol in the objective function which
    // is non-dummy and has a coefficient less than zero. If no symbol meets
    // the criteria, it means the objective function is at a minimum and no
    // symbol is returned.
    fn getEnteringSymbol(self: Solver, row: *const Row) ?Symbol {
        var it = row.cells.iterator();
        while (it.next()) |entry| {
            if (entry.key.tp != .Dummy and entry.value < 0.0) {
                return entry.key;
            }
        }
        return null;
    }

    // Compute the entering symbol for the dual optimize operation.
    //
    // This method will return the symbol in the row which has a positive
    // coefficient and yields the minimum ratio for its respective symbol
    // in the objective function. The provided row *must* be infeasible.
    // If no symbol is found which meats the criteria, no symbol is returned.
    fn getDualEnteringSymbol(self: Solver, row: *const Row) ?Symbol {
        var ratio: f32 = std.math.f32_max;
        var symbol: ?Symbol = null;
        var it = row.cells.iterator();
        while (it.next()) |entry| {
            if (entry.value > 0.0 and entry.key.tp != .Dummy) {
                const coeff = self.objective.coefficientFor(entry.key);
                const r = coeff/entry.value;
                if (r < ratio) {
                    ratio = r;
                    symbol = entry.key;
                }
            }
        }
        return symbol;
    }

    // Get the first Slack or Error symbol in the row.
    // If no such symbol is present, no nsymbol is returned.
    fn getAnyPivotableSymbol(self: Solver, row: *const Row) ?Symbol {
        var it = row.cells.iterator();
        while (it.next()) |entry| {
            if (entry.key.tp == .Slack or entry.key.tp == .Error) {
                return entry.key;
            }
        }
        return null;
    }

    // Compute the row which holds the exit symbol for a pivot.
    //
    // This method will return an iterator to the row in the row map
    // which holds the exit symbol. If no appropriate exit symbol is
    // found, null is returned. This indicates that
    // the objective function is unbounded.
    fn getLeavingRow(self: Solver, entering: Symbol) ?*RowMap.KV {
        var ratio: f32 = std.math.f32_max;
        var result: ?*RowMap.KV = null;

        var it = self.rows.iterator();
        while (it.next()) |entry| {
            if (entry.key.tp != .External) {
                const row = entry.value;
                const coeff = row.coefficientFor(entering);
                if (coeff < 0.0) {
                    const r = -row.constant / coeff;
                    if (r < ratio) {
                        ratio = r;
                        result = entry;
                    }
                }
            }
        }

        return result;
    }

    // Compute the leaving row for a marker variable.
    //
    // This method will return an iterator to the row in the row map
    // which holds the given marker variable. The row will be chosen
    // according to the following precedence:
    //
    // 1) The row with a restricted basic varible and a negative coefficient
    //    for the marker with the smallest ratio of -constant / coefficient.
    //
    // 2) The row with a restricted basic variable and the smallest ratio
    //    of constant / coefficient.
    //
    // 3) The last unrestricted row which contains the marker.
    //
    // If the marker does not exist in any row, null will be returned.
    // This indicates an internal solver error since
    // the marker *should* exist somewhere in the tableau.
    fn getMarkerLeavingRow(self: Solver, marker: Symbol) ?*RowMap.KV {
        var r1: f32 = std.math.f32_max;
        var r2: f32 = std.math.f32_max;
        var first: ?*RowMap.KV = null;
        var second: ?*RowMap.KV = null;
        var third: ?*RowMap.KV = null;

        var it = self.rows.iterator();
        while (it.next()) |entry| {
            const row = entry.value;
            const c = row.coefficientFor(marker);
            if (c == 0) continue;

            if (entry.key.tp == .External) {
                third = entry;
            } else if (c < 0.0) {
                const r = -row.constant / c;
                if (r < r1) {
                    r1 = r;
                    first = entry;
                }
            } else {
                const r = row.constant / c;
                if (r < r2) {
                    r2 = r;
                    second = entry;
                }
            }
        }
        if (first) |result| return result;
        if (second) |result| return result;
        return third;
    }

    // Remove the effects of a constraint on the objective function.
    fn removeConstraintEffects(self: *Solver, cn: *const Constraint, tag: Tag) !void {
        if (tag.marker.tp == .Error) {
            try self.removeMarkerEffects(tag.marker, cn.strength);
        }
        if (tag.other.tp == .Error) {
            try self.removeMarkerEffects(tag.other, cn.strength);
        }
    }

    // Remove the effects of an error marker on the objective function.
    fn removeMarkerEffects(self: *Solver, marker: Symbol, strength: f32) !void {
        if (self.rows.getValue(marker)) |*row| {
            try self.objective.insertRow(row, -strength);
        } else {
            try self.objective.insertSymbol(marker, -strength);
        }
    }


    pub fn dumps(self: Solver) void {
        std.debug.warn("\nObjective\n---------\n", .{});
        self.objective.dumps();

        std.debug.warn("\nTableau\n--------\n", .{});
        var rows = self.rows.iterator();
        while (rows.next()) |entry| {
            entry.key.dumps();
            std.debug.warn(" | ", .{});
            entry.value.dumps();
        }

        std.debug.warn("\nInfeasible\n----------\n", .{});
        for (self.infeasible_rows.items) |sym| {
            sym.dumps();
            std.debug.warn("\n", .{});
        }

        std.debug.warn("\nVariables\n---------\n", .{});
        var vars = self.vars.iterator();
        while (vars.next()) |entry| {
            std.debug.warn("{} = ", .{entry.key.name});
            entry.value.dumps();
            std.debug.warn("\n", .{});
        }

        std.debug.warn("\nEdit Variables\n--------------\n", .{});
        var edits = self.edits.iterator();
        while (edits.next()) |entry| {
            std.debug.warn("{} = {}\n", .{entry.key, entry.value});
        }

        std.debug.warn("\nConstraints\n-----------\n", .{});
        var cns = self.cns.iterator();
        while (cns.next()) |entry| {
            entry.key.dumps();
        }
        std.debug.warn("\n", .{});
    }

};

test "strength" {
    const testing = std.testing;
    const mmedium = Strength.createWeighted(0, 1, 0, 1.25);
    testing.expectEqual(mmedium, 1250.0);
    const smedium = Strength.create(0, 100, 0);
    testing.expectEqual(smedium, 100000.0);
}

test "variables" {
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var v1 = Variable{.name="v1", .value=1.0};
    var v2 = Variable{.name="v2", .value=2.0};

    var t = v1.invert();
    testing.expectEqual(t.coefficient, -1);

    t = v1.mul(3);
    testing.expectEqual(t.coefficient, 3);

    t = v1.div(2);
    testing.expectEqual(t.coefficient, 0.5);

    testing.expectEqual(t.variable, &v1);

    // Adding variables returns an expression
    var expr = try v1.add(allocator, &v2);
    testing.expectEqual(expr.terms.items.len, 2);
    testing.expectEqual(expr.value(), 3.0);
    expr.deinit();

    expr = try v1.sub(allocator, &v2);
    testing.expectEqual(expr.terms.items.len, 2);
    testing.expectEqual(expr.value(), -1.0);
    expr.deinit();
}

test "terms" {
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var x = Variable{.name="x", .value=2.0};
    var t1 = Term{.variable=&x};

    var t2 = t1.mul(3);
    testing.expectEqual(t2.coefficient, 3.0);
    testing.expectEqual(t2.variable, t1.variable);
    testing.expectEqual(t2.value(), 6.0);

    t2 = t1.div(2);
    testing.expectEqual(t2.coefficient, 0.5);
    testing.expectEqual(t2.value(), 1.0);

    t2 = t1.invert();
    testing.expectEqual(t2.coefficient, -1);
    testing.expectEqual(t2.value(), -2.0);

    // Adding variables returns an expression
    var expr = try t1.add(allocator, 3);
    testing.expectEqual(expr.terms.items.len, 1);
    testing.expectEqual(expr.value(), 5.0);
    expr.deinit();

    expr = try t1.add(allocator, &t2);
    testing.expectEqual(expr.value(), 0.0);
    expr.deinit();
}

test "expressions" {
    const allocator = std.heap.page_allocator;
    const testing = std.testing;
    var x = Variable{.name="x", .value=1.0};
    var y = Variable{.name="y", .value=3.0};
    var z = Variable{.name="z", .value=2.0};
    var expr = try Expression.init(allocator, &[_]Term{
        Term{.variable=&x},
        Term{.variable=&y},
    });
    testing.expectEqual(expr.terms.items.len, 2);
    testing.expectEqual(expr.value(), 4.0);
    expr.deinit();

    expr = try Expression.init(allocator, &[_]Term{
        x.mul(2),
        x.mul(4),
    });
    defer expr.deinit();
    testing.expectEqual(expr.terms.items.len, 1);
    testing.expectEqual(expr.value(), 6.0);

    // 6x * 3
    var expr2 = try expr.mul(allocator, 3);
    testing.expectEqual(expr2.value(), 18.0);
    expr2.deinit();

    // 6x / 2
    expr2 = try expr.div(allocator, 2);
    testing.expectEqual(expr2.value(), 3.0);
    expr2.deinit();

    // -6x
    expr2 = try expr.invert(allocator);
    testing.expectEqual(expr2.value(), -6.0);
    expr2.deinit();

    // 6x - 6
    expr2 = try expr.add(allocator, -6);
    testing.expectEqual(expr2.value(), 0.0);
    expr2.deinit();

    // 6x + y
    expr2 = try expr.add(allocator, &y);
    testing.expectEqual(expr2.value(), 9.0);
    expr2.deinit();

    // 6x + 3y
    expr2 = try expr.add(allocator, &y.mul(3));
    testing.expectEqual(expr2.value(), 15.0);
    expr2.deinit();

    // 6x - y
    expr2 = try expr.sub(allocator, &y);
    testing.expectEqual(expr2.value(), 3.0);
    //expr2.deinit();

    // 6x + 6x - y
    var expr3 = try expr.add(allocator, expr2);
    testing.expectEqual(expr3.terms.items.len, 2);
    testing.expectEqual(expr3.value(), 9.0);
    //expr3.deinit();

    var expr4 = try expr3.add(allocator, &z);
    testing.expectEqual(expr4.terms.items.len, 3);
    testing.expectEqual(expr4.value(), 11.0);
}

test "constraints" {
    // TODO:

}



test "row" {
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var row = Row.init(allocator);
    testing.expectEqual(row.add(2.0), 2.0);
    row.reverseSign();
    testing.expectEqual(row.constant, -2.0);
    testing.expectEqual(row.add(2.0), 0.0);


    var e1 = Symbol{.id=1, .tp=.Error};
    var e2 = Symbol{.id=2, .tp=.Error};
    var e3 = Symbol{.id=3, .tp=.Error};

    // Add symbols
    try row.insertSymbol(e1, 2.0);
    try row.insertSymbol(e2, 1.0);
    testing.expectEqual(row.coefficientFor(e1), 2.0);
    testing.expectEqual(row.coefficientFor(e2), 1.0);

    // Re-add with reversed coeff, it should cancel
    try row.insertSymbol(e1, -2.0);
    testing.expectEqual(row.cells.get(e1), null);

    try row.insertSymbol(e1, 2.0);
    testing.expectEqual(row.coefficientFor(e1), 2.0);
    testing.expectEqual(row.coefficientFor(e2), 1.0);


    testing.expectError(error.CannotSolveForUnknownSymbol,
                        row.solveForSymbol(e3));

    // 2 + 2e1 + 1e2 = e3 --> e1 = e3 / 2 - e2 / 2 - 1
    _ = row.add(2);
    try row.solveFor(e3, e1);
    testing.expectEqual(row.cells.get(e1), null); // Removes e1
    testing.expectEqual(row.coefficientFor(e1), 0.0);
    testing.expectEqual(row.coefficientFor(e2), -0.5);
    testing.expectEqual(row.coefficientFor(e3), 0.5);
    testing.expectEqual(row.constant, -1.0);

    row.reverseSign();
    testing.expectEqual(row.coefficientFor(e2), 0.5);
    testing.expectEqual(row.coefficientFor(e3), -0.5);
    testing.expectEqual(row.constant, 1.0);

}


test "solver-variable-managment" {
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var solver = Solver.init(allocator);
    defer solver.deinit();
    var v1 = Variable{.name="v1"};
    var v2 = Variable{.name="v2"};

    testing.expect(!solver.hasVariable(&v1));

    try solver.addVariable(&v1, Strength.medium);
    testing.expectEqual(solver.cns.size, 1);
    testing.expectEqual(solver.edits.size, 1);
    testing.expectEqual(solver.rows.size, 1);
    testing.expect(solver.edits.contains(&v1));

    testing.expect(solver.hasVariable(&v1));

    // Already theres
    testing.expectError(error.DuplicateVariable,
        solver.addVariable(&v1, Strength.medium));

    // Editable variables cannot have strength == required
    testing.expectError(error.BadRequiredStrength,
        solver.addVariable(&v2, Strength.required));
    testing.expectEqual(solver.cns.size, 1); // Should still be 1

    // Not yet added
    testing.expect(!solver.hasVariable(&v2));

    // Cant suggest a value that wasn't added
    testing.expectError(error.UnknownVariable,
        solver.suggestValue(&v2, 5.0));
    testing.expectError(error.UnknownVariable,
        solver.removeVariable(&v2));

    try solver.addVariable(&v2, Strength.medium);
    testing.expect(solver.hasVariable(&v2));
    testing.expectEqual(solver.cns.size, 2);

    try solver.removeVariable(&v1);
    testing.expect(!solver.hasVariable(&v1));

    solver.reset();
    testing.expect(!solver.hasVariable(&v2));
}

test "suggestions-medium-suggestion-overrides-weak-constraint" {
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var solver = Solver.init(allocator);
    defer solver.deinit();
    var v1 = Variable{.name="v1"};

    // This builds a constraint
    try solver.addVariable(&v1, Strength.medium);

    var c = try solver.buildConstraint(&v1, .eq, 1, Strength.weak);
    defer c.deinit();
    try solver.addConstraint(&c);
    testing.expectEqual(solver.cns.size, 2);
    testing.expect(solver.hasConstraint(&c));
    testing.expectError(error.DuplicateConstraint, solver.addConstraint(&c));

    try solver.suggestValue(&v1, 2);
    //testing.expectEqual(@as(f32, 2.0), solver.edits.getValue(&v1).?.constant);

    solver.updateVariables();

    //solver.dumps();

    // Since v1 is medium it overwrites the weak constraint v1 == 1
    testing.expectEqual(@as(f32, 2.0), v1.value);
}

test "suggestions-weak-suggestion-is-overriden-by-medium-constraint" {
    //if (!@hasDecl(std, "crap")) return error.SkipZigTest;
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var solver = Solver.init(allocator);
    var v1 = Variable{.name="v1"};

    try solver.addVariable(&v1, Strength.weak);

    var c = try solver.buildConstraint(&v1, .eq, 4, Strength.medium);
    defer c.deinit();
    try solver.addConstraint(&c);
    try solver.suggestValue(&v1, 2);

    solver.updateVariables();

    //solver.dumps();

    // Since v1 is weak is is now overridden by the medium constraint v1 == 1
    testing.expectEqual(@as(f32, 4.0), v1.value);
}

test "suggestions-readme-example" {
    //if (!@hasDecl(std, "crap")) return error.SkipZigTest;
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var solver = Solver.init(allocator);
    defer solver.deinit();

    var width = Variable{.name="width"};
    var height = Variable{.name="height"};

    try solver.addVariable(&width, Strength.medium);
    try solver.addVariable(&height, Strength.medium);

    // 16 * width = 9 * height
    var aspect = try solver.buildConstraint(
        &width.mul(3), .eq, &height.mul(4), Strength.strong);
    defer aspect.deinit();
    try solver.addConstraint(&aspect);

    var min_width = try solver.buildConstraint(
        &width, .gte, 320, Strength.strong);
    defer min_width.deinit();
    try solver.addConstraint(&min_width);

    try solver.suggestValue(&height, 1);
    solver.dumps();

    solver.updateVariables(); // Updates the interal value of every variable

    testing.expectEqual(width.value, 320);
    testing.expectEqual(height.value, 240);
}

test "suggestions-3" {
    //if (!@hasDecl(std, "crap")) return error.SkipZigTest;
    const testing = std.testing;
    const allocator = std.heap.page_allocator;
    var solver = Solver.init(allocator);
    var v1 = Variable{.name="v1"};
    var v2 = Variable{.name="v2"};

    try solver.addVariable(&v1, Strength.medium);
    try solver.addVariable(&v2, Strength.medium);
    var c = try solver.buildConstraint(&v1, .gte, 1, Strength.weak);
    defer c.deinit();
    try solver.addConstraint(&c);

    var expr = try v1.add(allocator, 10);
    defer expr.deinit();

    // v2 >= v1 + 10
    var c2 = try solver.buildConstraint(&v2, .gte, expr, Strength.strong);
    defer c2.deinit();
    try solver.addConstraint(&c2);

    try solver.suggestValue(&v1, 2);

    solver.updateVariables();
    solver.dumps();
    std.debug.warn("v1 = {}\n", .{v1.value});
    std.debug.warn("v2 = {}\n", .{v2.value});
    testing.expect(v1.value >= 1);
    testing.expect(v2.value >= v1.value + 10);

}

// test "benchmark" {
//     TODO:
//     const testing = std.testing;
//     var buf: [10000]u8 = undefined;
//     const allocator = &std.heap.FixedBufferAllocator.init(&buf).allocator;
//     var solver = Solver.init(allocator);
//
//     Create custom strength
//     const mmedium = Strength.createWeighted(0, 1, 0, 1.25);
//     testing.expectEqual(mmedium, 1250.0);
//     const smedium = Strength.create(0, 100, 0);
//     testing.expectEqual(smedium, 100000.0);
//
//     Create some variables
//     var left = Variable{.name="left"};
//     var height = Variable{.name="height"};
//     var top = Variable{.name="top"};
//     var width = Variable{.name="width"};
//     var contents_top = Variable{.name="contents_top"};
//     var contents_bottom = Variable{.name="contents_bottom"};
//     var contents_left = Variable{.name="contents_left"};
//     var contents_right = Variable{.name="contents_right"};
//     var midline = Variable{.name="midline"};
//     var ctleft = Variable{.name="ctleft"};
//     var ctheight = Variable{.name="ctheight"};
//     var cttop = Variable{.name="cttop"};
//     var ctwidth = Variable{.name="ctwidth"};
//     var lb1left = Variable{.name="lb1left"};
//     var lb1height = Variable{.name="lb1height"};
//     var lb1top = Variable{.name="lb1top"};
//     var lb1width = Variable{.name="lb1width"};
//     var lb2left = Variable{.name="lb2left"};
//     var lb2height = Variable{.name="lb2height"};
//     var lb2top = Variable{.name="lb2top"};
//     var lb2width = Variable{.name="lb2width"};
//     var lb3left = Variable{.name="lb3left"};
//     var lb3height = Variable{.name="lb3height"};
//     var lb3top = Variable{.name="lb3top"};
//     var lb3width = Variable{.name="lb3width"};
//     var fl1left = Variable{.name="fl1left"};
//     var fl1height = Variable{.name="fl1height"};
//     var fl1top = Variable{.name="fl1top"};
//     var fl1width = Variable{.name="fl1width"};
//     var fl2left = Variable{.name="fl2left"};
//     var fl2height = Variable{.name="fl2height"};
//     var fl2top = Variable{.name="fl2top"};
//     var fl2width = Variable{.name="fl2width"};
//     var fl3left = Variable{.name="fl3left"};
//     var fl3height = Variable{.name="fl3height"};
//     var fl3top = Variable{.name="fl3top"};
//     var fl3width = Variable{.name="fl3width"};
//
//     Add the edit variables
//     try solver.addVariable(&width, Strength.strong);
//     try solver.addVariable(&height, Strength.strong);
//
// }
