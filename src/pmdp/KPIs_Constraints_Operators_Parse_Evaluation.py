import math
import re
import decimal
import numpy as np
from collections import deque


# Define operator precedence, associativity, and descriptions
OPERATOR_INFO = {
    '(': {'priority': 0, 'associativity': 'left', 'category': ['KPI', 'HC', 'SC'], 'description': 'Parentheses'},
    ')': {'priority': 0, 'associativity': 'left', 'category': ['KPI', 'HC', 'SC'], 'description': 'Parentheses'},
    '[': {'priority': 1, 'associativity': 'left', 'category': ['KPI'], 'description': 'Vector Parentheses'},
    ']': {'priority': 1, 'associativity': 'left', 'category': ['KPI'], 'description': 'Vector Parentheses'},
    '!': {'priority': 2, 'associativity': 'right', 'category': ['HC', 'SC'], 'description': 'Logical Not'},
    r'\mm': {'priority': 2, 'associativity': 'right', 'category': ['KPI'],
             'description': r'Dynamic max-min normalization (e.g., \mm x or \mm(x))'},
    r'\RB': {'priority': 2, 'associativity': 'right', 'category': ['KPI'],
             'description': r'Select Maximum from Set (e.g., \RB[1,2,3])'},
    r'\LB': {'priority': 2, 'associativity': 'right', 'category': ['KPI'],
             'description': r'Select Minimum from Set (e.g., \LB[1,2,3])'},
    r'\max': {'priority': 2, 'associativity': 'right', 'category': ['SC'], 'description': 'Maximizing'},
    r'\min': {'priority': 2, 'associativity': 'right', 'category': ['SC'], 'description': 'Minimizing'},
    r'\log': {'priority': 2, 'associativity': 'right', 'category': ['KPI'], 'description': 'Logarithm'},
    r'\abs': {'priority': 2, 'associativity': 'right', 'category': ['KPI'], 'description': 'Absolute Value'},
    r'\sqrt': {'priority': 2, 'associativity': 'right', 'category': ['KPI'],
               'description': r'Square root or n-th root operator. Format: \sqrt{value} or \sqrt[n]{value} (default n=2)'},
    r'\times': {'priority': 3, 'associativity': 'left', 'category': ['KPI'], 'description': 'Multiplication'},
    r'\div': {'priority': 3, 'associativity': 'left', 'category': ['KPI'], 'description': 'Division'},
    r'\cdot': {'priority': 3, 'associativity': 'left', 'category': ['KPI'], 'description': 'Vector Dot Product'},
    '+': {'priority': 4, 'associativity': 'left', 'category': ['KPI'], 'description': 'Addition'},
    '-': {'priority': 4, 'associativity': 'left', 'category': ['KPI'], 'description': 'Subtraction'},
    '>': {'priority': 5, 'associativity': 'left', 'category': ['HC', 'SC'], 'description': 'Greater Than'},
    r'\ge': {'priority': 5, 'associativity': 'left', 'category': ['HC', 'SC'],
             'description': 'Greater Than or Equal To'},
    '<': {'priority': 5, 'associativity': 'left', 'category': ['HC', 'SC'], 'description': 'Less Than'},
    r'\le': {'priority': 5, 'associativity': 'left', 'category': ['HC', 'SC'],
             'description': 'Less Than or Equal To'},
    '=': {'priority': 6, 'associativity': 'left', 'category': ['HC', 'SC'], 'description': 'Equal To'},
    r'\ne': {'priority': 6, 'associativity': 'left', 'category': ['HC', 'SC'], 'description': 'Not Equal To'},
    r'\wedge': {'priority': 7, 'associativity': 'left', 'category': ['HC', 'SC'], 'description': 'Logical AND'},
    r'\vee': {'priority': 8, 'associativity': 'left', 'category': ['HC', 'SC'], 'description': 'Logical OR'},
    ',': {'priority': 9, 'associativity': 'left', 'category': ['KPI'], 'description': 'Comma Operator'}
}

def preprocess_tokens(tokens):
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Check if it's a minus sign followed by a number
        if token == '-' and i + 1 < len(tokens) and re.match(r'^-?\d+(\.\d+)?$', tokens[i + 1]):
            # Check the element before the minus sign
            if i == 0 or tokens[i - 1] in OPERATOR_INFO or tokens[i - 1] == '(':
                # Combine the minus sign and number into a negative number
                combined_number = token + tokens[i + 1]
                processed_tokens.append(combined_number)
                i += 2  # Skip the next number
                continue
        processed_tokens.append(token)
        i += 1
    return processed_tokens

class ExpressionParser:
    def __init__(self, operator_info):
        self.operator_info = operator_info
        self.expression = ''

    def tokenize(self, expression, category='KPI'):
        self.expression = expression
        # Define regular expression patterns
        pattern = re.compile(
            r'\[.*?\]'                    # Match content within square brackets
            r'|\\[a-zA-Z]+'               # Match backslash followed by a letter
            r'|(?<!\w)-?\d+\.\d+'         # Match decimal numbers, allowing for negatives
            r'|(?<!\w)-?\d+'              # Match integer numbers, allowing for negatives
            r'|\w+'                       # Match variable names
            r'|[+\-*/^(),]'               # Match operators and commas
            r'|[^\s]'                     # Match other non-whitespace characters
        )
        matches = pattern.findall(expression)
        matches = preprocess_tokens(matches)

        if '~' in expression:
            # Handle cases like 'a~b'
            i = 0
            while i < len(matches):
                if matches[i] == '~' and i > 0 and i < len(matches) - 1:
                    combined_token = matches[i - 1] + matches[i] + matches[i + 1]
                    matches = matches[:i - 1] + [combined_token] + matches[i + 2:]
                    i -= 1  # Adjust index to continue checking merged list
                else:
                    i += 1

        # Validate that operators belong to the correct category
        illegal_operators = []
        for match in matches:
            if match in self.operator_info and category not in self.operator_info[match]['category']:
                illegal_operators.append(match)

        if illegal_operators:
            error_message = f"{expression} contains illegal operators for {category} expressions: {', '.join(illegal_operators)}."
            error_message += " Details: "
            error_message += ", ".join(
                [f"{op} (belongs to {self.operator_info[op]['category']})" for op in illegal_operators])
            raise ValueError(error_message)

        return matches

    def parse(self, expression, category='KPI'):
        tokens = self.tokenize(expression, category)
        output_queue = deque()
        operator_stack = []
        for token in tokens:
            if re.match(r'-?\d+(\.\d+)?', token) or re.match(r'\[.*?\]', token) or token.isalnum() or re.match(
                    r'([a-zA-Z0-9_]+)~([a-zA-Z0-9_])+', token) or re.match(r'[a-zA-Z0-9_]+', token):
                output_queue.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()  # Pop left parenthesis
            elif token in self.operator_info and category in self.operator_info[token]['category']:
                if token in [r'\RB', r'\LB', r'\mm', r'\max', r'\min', r'\log', r'\abs', r'\sqrt']:
                    operator_stack.append(token)
                else:
                    while (operator_stack and operator_stack[-1] in self.operator_info and
                           operator_stack[-1] != '(' and
                           ((self.operator_info[token]['associativity'] == 'left' and
                             self.operator_info[token]['priority'] >= self.operator_info[operator_stack[-1]]['priority']) or
                            (self.operator_info[token]['associativity'] == 'right' and
                             self.operator_info[token]['priority'] > self.operator_info[operator_stack[-1]]['priority']))):
                        output_queue.append(operator_stack.pop())
                    operator_stack.append(token)

        while operator_stack:
            top = operator_stack.pop()
            if top == '(':
                raise ValueError("Mismatched parentheses")
            output_queue.append(top)

        return self.build_ast(output_queue)

    def build_ast(self, output_queue):
        stack = []

        while output_queue:
            token = output_queue.popleft()
            if token not in self.operator_info and (
                    re.match(r'-?\d+(\.\d+)?', token) or re.match(r'\[.*?\]', token) or token.isalnum()) or re.match(
                r'([a-zA-Z0-9_]+)~([a-zA-Z0-9_]+)', token) or re.match(r'[a-zA-Z0-9_]+', token):
                stack.append(token)
            elif token in self.operator_info:
                operator = token
                if operator in [r'\RB', r'\LB', r'\mm', r'\max', r'\min', r'\log', r'\abs', r'\sqrt']:
                    # Handle unary or multi-ary operators, with special handling for \sqrt
                    if operator == r'\sqrt':
                        # Check for degree parameter in brackets, e.g., "[n]"
                        if len(stack) == 2:
                            degree_token = stack[0]
                            # Remove brackets to get degree value string
                            degree_str = degree_token[1:-1].strip()
                            try:
                                degree_value = float(degree_str)
                            except ValueError:
                                raise ValueError(f"Invalid degree value in {degree_token}")
                            if not stack:
                                raise ValueError(f"Error building AST: insufficient operands for operator {operator}")
                            operand = stack.pop()
                            stack.pop()  # Pop degree parameter
                            # Construct binary AST：(\sqrt, degree, operand)
                            stack.append((operator, degree_value, operand))
                        else:
                            # If there is no degree parameter, it is a unary case: \sqrt value
                            if not stack:
                                raise ValueError(f"Error building AST: insufficient operands for operator {operator}")
                            operand = stack.pop()
                            
                            stack.append((operator, operand))
                    else:
                        # Handle other unary or multi-ary operators (e.g., \log may have two operands)
                        operands = []
                        if stack:
                            operands.append(stack.pop())
                        else:
                            raise ValueError(f"Error building AST: insufficient operands for operator {operator}")

                        # Handle \log operator: if there is a comma, take the second operand (e.g., \log x , base)
                        if operator == r'\log' and output_queue and output_queue[0] == ',':
                            output_queue.popleft()  # Remove comma
                            if stack:
                                operands.insert(0, stack.pop())
                            else:
                                raise ValueError(f"Error building AST: insufficient operands for operator {operator}")
                        stack.append((operator, *operands))
                else:
                    # Handle binary operators
                    if len(stack) < 2:
                        raise ValueError(f"Error building AST: insufficient operands for operator {token}")
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    stack.append((token, operand1, operand2))

        if len(stack) != 1:
            raise ValueError(f"Error building AST: {stack}")

        return stack[0]

class ExpressionEvaluator:
    def __init__(self, context={}):
        self.context = context

    def evaluate(self, ast, dynamic_bounds={}):
        if isinstance(ast, tuple):
            operator = ast[0]
            if len(ast) == 2:  # one operand operator
                if operator not in [r'\mm', r'\max', r'\min', r'\log', r'\abs', r'\sqrt']:
                    operand = self.evaluate(ast[1], dynamic_bounds)[0]
                    return self.apply_operator(operator, operand, dynamic_bounds=dynamic_bounds), dynamic_bounds
                else:
                    # Handle operators that require special processing
                    operand = ast[1]
                    return self.apply_operator(operator, operand, dynamic_bounds=dynamic_bounds), dynamic_bounds
            elif len(ast) >= 3:  # binary or multi-ary operators
                operands = []
                for operand_ast in ast[1:]:
                    operand = self.evaluate(operand_ast, dynamic_bounds)[0]
                    operands.append(operand)
                return self.apply_operator(operator, *operands, dynamic_bounds=dynamic_bounds), dynamic_bounds
        else:
            try:

                # 常量
                if isinstance(ast, (int, float)):
                    ast = str(ast)
                if re.match(r'-?\d+(\.\d+)?', ast):
                    return eval(ast), dynamic_bounds
                # Check if it is a vector
                if ast.startswith('[') and ast.endswith(']'):
                    new_ast = ''
                    values = ast.replace('[', '').replace(']', '').split(',')
                    new_ast += '['
                    for value in values:
                        value = value.strip()
                        if re.match(r'-?\d+(\.\d+)?', value):
                            new_ast += f"{value},"
                        elif value in self.context:
                            actual_value = self.context[value]
                            if actual_value == 'NotSelected':
                                actual_value = 0
                            new_ast += f"{actual_value},"
                        elif any([op in value for op in OPERATOR_INFO.keys()]):
                            # If the value is an expression, recursively evaluate it
                            parser = ExpressionParser(OPERATOR_INFO)
                            subKPI_ast = parser.parse(value)
                            actual_value = self.evaluate(subKPI_ast, dynamic_bounds)[0]
                            new_ast += f"{actual_value},"
                        else:
                            raise ValueError(f"The value '{value}' is not defined in the context")
                    new_ast = new_ast.rstrip(',')  # Remove trailing comma
                    new_ast += ']'
                    return eval(new_ast), dynamic_bounds

                # Get variable value from context
                value = self.context[ast]
                if value == 'NotSelected':
                    value = 0

                # Check if it is a vector
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    return eval(value), dynamic_bounds

                return value, dynamic_bounds

            except (ValueError, KeyError):
                return self.context[ast], dynamic_bounds

    def apply_operator(self, operator, *operands, dynamic_bounds={}):
        if 'NotSelected' in operands:
            operands = tuple(0 if op == 'NotSelected' else op for op in operands)

        if operator == '+':
            if all(isinstance(op, list) for op in operands):
                return [sum(values) for values in zip(*operands)]
            else:
                return sum(operands)
        elif operator == '-':
            if all(isinstance(op, list) for op in operands):
                return [a - sum(operands[1:]) for a in operands[0]]
            else:
                result = operands[0]
                for op in operands[1:]:
                    result -= op
                return result
        elif operator == r'\times':
            result = operands[0]
            for op in operands[1:]:
                result *= op
            return result
        elif operator == r'\div':
            result = operands[0]
            for op in operands[1:]:
                result /= op
            return result
        elif operator == r'\cdot':
            # Vector dot product
            return sum(a * b for a, b in zip(operands[0], operands[1]))
        elif operator == r'\RB':
            if not isinstance(operands[0], (list, set)):
                raise ValueError(f"Unsupported operand for RB: {operands[0]}")
            return max(operands[0])
        elif operator == r'\LB':
            if not isinstance(operands[0], (list, set)):
                raise ValueError(f"Unsupported operand for LB: {operands[0]}")
            return min(operands[0])
        elif operator == r'\mm':
            var_name = operands[0]
            value = self.context[var_name]
            if value == 'NotSelected':
                value = 0
            self.update_dynamic_bounds(var_name, value, dynamic_bounds)
            if var_name in dynamic_bounds:
                min_val, max_val = dynamic_bounds[var_name]
                direction = self.context.get(f"{var_name}_direction", "positive")
                if max_val != min_val:
                    if direction == "positive":
                        return (value - min_val) / (max_val - min_val)
                    else:
                        return (max_val - value) / (max_val - min_val)
                else:
                    return 1.0
            else:
                raise ValueError(f"Dynamic bounds not defined for variable: {var_name}")
        elif operator == '!':
            return not operands[0]
        elif operator == '>':
            return operands[0] > operands[1]
        elif operator == r'\ge':
            return operands[0] >= operands[1]
        elif operator == '<':
            return operands[0] < operands[1]
        elif operator == r'\le':
            return operands[0] <= operands[1]
        elif operator == '=':
            return operands[0] == operands[1]
        elif operator == r'\ne':
            return operands[0] != operands[1]
        elif operator == r'\wedge':
            return all(operands)
        elif operator == r'\vee':
            return any(operands)
        elif operator == r'\max':
            var_name = operands[0]
            value = self.context[var_name]
            if value == 'NotSelected':
                value = 0
            self.update_dynamic_bounds(var_name, value, dynamic_bounds)
            if var_name in dynamic_bounds:
                min_val, max_val = dynamic_bounds[var_name]
                if max_val != min_val:
                    return (value - min_val) / (max_val - min_val)
                else:
                    return 1.0
            else:
                raise ValueError(f"Dynamic bounds not defined for variable: {var_name}")
        elif operator == r'\min':
            var_name = operands[0]
            value = self.context[var_name]
            if value == 'NotSelected':
                value = 0
            self.update_dynamic_bounds(var_name, value, dynamic_bounds)
            if var_name in dynamic_bounds:
                min_val, max_val = dynamic_bounds[var_name]
                if max_val != min_val:
                    return (max_val - value) / (max_val - min_val)
                else:
                    return 1.0
            else:
                raise ValueError(f"Dynamic bounds not defined for variable: {var_name}")
        elif operator == r'\log':
            if len(operands) == 1:
                if operands[0] in self.context:
                    value = self.context[operands[0]]
                else:
                    value = operands[0]
                base = math.e  # 默认自然对数
            elif len(operands) == 2:
                if operands[0] in self.context:
                    value = self.context[operands[0]]
                else:
                    value = operands[0]
                if operands[1] in self.context:
                    base = self.context[operands[1]]
                else:
                    base = operands[1]
                value, base = operands
            else:
                raise ValueError(r"\log operator takes one or two operands.")
            if value <= 0 or base <= 0 or base == 1:
                raise ValueError("Invalid input for logarithm.")
            return math.log(value, base)
        elif operator == r'\abs':
            if operands[0] in self.context:
                value = self.context[operands[0]]
                return abs(value)
            else:
                return abs(operands[0])
            
        elif operator == r'\sqrt':
    # Handle square root operator, supports \sqrt{value} or \sqrt[n]{value} format
            if len(operands) == 1:
                if operands[0] in self.context:
                    value = self.context[operands[0]]
                else:
                    value = float(operands[0])
                degree = 2
            elif len(operands) == 2:
                degree_operand = operands[0]
                value = operands[1]
                # Handle degree: if it is a vector with only one element, extract that element
                if isinstance(degree_operand, list):
                    if len(degree_operand) != 1:
                        raise ValueError(f"Degree operand for sqrt must be a single value, got {degree_operand}")
                    degree = degree_operand[0]
                else:
                    degree = degree_operand
            else:
                raise ValueError(r"\sqrt operator takes one or two operands.")

            # Check numerical validity
            if degree == 0:
                raise ValueError("Degree cannot be zero in root calculation.")
            # Handle negative even roots
            if isinstance(value, (int, float)) and value < 0 and isinstance(degree, (int, float)) and degree % 2 == 0:
                raise ValueError(f"Cannot take even root of negative number: {value}^{1/degree}")
       
            return math.pow(value, 1.0 / degree)

        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def update_dynamic_bounds(self, var_name, value, dynamic_bounds={}):
        if value == 'NotSelected':
            value = 0
        if var_name not in dynamic_bounds:
            dynamic_bounds[var_name] = (value, value)
        else:
            min_val, max_val = dynamic_bounds[var_name]
            if value < min_val:
                min_val = value
            if value > max_val:
                max_val = value
            dynamic_bounds[var_name] = (min_val, max_val)

