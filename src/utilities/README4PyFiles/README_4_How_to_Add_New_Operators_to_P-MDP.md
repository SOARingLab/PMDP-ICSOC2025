# Readme for How_to_Add_New_Operators_to_P-MDP in KPIs_Constraints_Operators_Parse_Evaluation.py

## Overview

`KPIs_Constraints_Operators_Parse_Evaluation.py` is a Python module designed to parse and evaluate expressions related to Key Performance Indicators (KPIs), Hard Constraints (HC), and Soft Constraints (SC). The module supports various operators for mathematical, logical, and normalization operations. This guide will provide instructions for adding new operators to the module.

## Structure

The module consists of the following main components:

1. **Global Variables:**

   - `dynamic_bounds`: A dictionary to store dynamic bounds for variables and KPIs that require the use of dymaic Max-Min Normalization.
   - `OPERATOR_INFO`: A dictionary defining operator priorities, associativity, categories, and descriptions.
2. **Classes:**

   - `ExpressionParser`: Parses the expressions and converts them into an Abstract Syntax Tree (AST).
   - `ExpressionEvaluator`: Evaluates the AST using the given context and dynamic bounds.
3. **Functions:**

   - `tokenize()`: Converts an expression into a list of tokens.
   - `parse()`: Converts tokens into an AST.
   - `build_ast()`: Builds the AST from the tokenized expression.
   - `evaluate()`: Evaluates the AST, that is, to calculate the result of KPI or Constraint.
   - `apply_operator()`: Applies the given operator to the operands.
   - `update_dynamic_bounds()`: Updates the dynamic bounds for a given variable and KPIs.

## Adding New Operators

### Unary Operators

To add a new unary operator:

1. **Update `OPERATOR_INFO`:**

   - Add a new entry for the operator with its priority, associativity, categories, and description.

   ```python
   'new_unary_operator': {'priority': X, 'associativity': 'right', 'category': ['KPI', 'HC', 'SC'], 'description': 'New Unary Operator Description'},
   ```
2. **Update `ExpressionParser.parse()`:**

   - Add the new unary operator to the parsing list for right associativity operators.

   ```python
   if token in ['RB', 'LB', 'mm', r'\max', r'\min', 'new_unary_operator']:  # Unary Operator
       ...
   ```
3. **Update `ExpressionParser.build_ast()`:**

   - Add the new unary operator to the list of recognized unary operators.

   ```python
   if token in ['RB', 'LB', 'mm', r'\max', r'\min', 'new_unary_operator']:  # Unary Operator
       ...
   ```
4. **Update `ExpressionEvaluator.apply_operator()`:**

   - Add the logic to handle the new unary operator.

   ```python
   elif operator == 'new_unary_operator':
       # Implement the logic for the new unary operator
       return ...
   ```

### Binary Operators

To add a new binary operator:

1. **Update `OPERATOR_INFO`:**

   - Add a new entry for the operator with its priority, associativity, categories, and description.

   ```python
   'new_binary_operator': {'priority': X, 'associativity': 'left', 'category': ['KPI', 'HC', 'SC'], 'description': 'New Binary Operator Description'},
   ```
2. **Update `ExpressionParser.build_ast()`:**

   - Ensure the new binary operator is handled correctly during AST construction. No changes needed here as binary operators are handled generically.
3. **Update `ExpressionEvaluator.apply_operator()`:**

   - Add the logic to handle the new binary operator.

   ```python
   elif operator == 'new_binary_operator':
       # Implement the logic for the new binary operator
       return ...
   ```

### N-ary Operators

N-ary operators can often be decomposed into multiple applications of binary and unary operators. Use combinations of unary and binary operators to achieve n-ary operations without defining special n-ary operators.

## Naming Conventions for Operators

When naming operators, adhere to these guidelines:

- Operator names must use a single symbol, e.g., `^` for exponential operator, or begin with a letter, backslash, or underscore and contain no special symbols.
- Avoid use special symbols with letters, backslashes, or underscores, such as hyphens `na^me`, which may conflict with tokenize() rules.
- Some examples of valid operator names: `^` for exponential operator, `exp` for exponent operator, `sqrt_3` for the 3rd square root, etc.

## Conclusion

By following the steps outlined above, you can easily extend the functionality of `KPIs_Constraints_Operators_Parse_Evaluation.py` to support additional unary, binary, or n-ary operators. Ensure to update the relevant sections in `OPERATOR_INFO`, `ExpressionParser`, and `ExpressionEvaluator` to handle the new operators appropriately.
