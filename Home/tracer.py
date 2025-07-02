import sys
import ast
import json
import traceback
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import builtins

# Universal event types to cover all basic Python logic
class EventType(Enum):
    START = "START"
    DEF = "DEF"
    CLASS = "CLASS"
    CALL = "CALL"
    ENTER = "ENTER"
    RETURN = "RETURN"
    ASSIGN = "ASSIGN"
    IF = "IF"
    LOOP = "LOOP"
    LOOP_ITERATION = "LOOP_ITERATION"
    EXCEPTION = "EXCEPTION"
    PRINT = "PRINT"
    EXPRESSION = "EXPRESSION"
    IMPORT = "IMPORT"
    WITH = "WITH"
    TRY = "TRY"
    EXCEPT = "EXCEPT"
    FINALLY = "FINALLY"
    MULTIPLICATION = "MULTIPLICATION"
    ADDITION = "ADDITION"
    SUBTRACTION = "SUBTRACTION"
    DIVISION = "DIVISION"
    COMPARISON = "COMPARISON"  # New event type for comparisons

@dataclass
class TraceEvent:
    step: int
    line: int
    event: EventType
    code: str
    variables: Dict[str, Any]
    stack: List[Dict[str, Any]]
    note: Optional[str] = None
    output: Optional[str] = None
    call_sequence: Optional[str] = None  # New field to track call sequence

@dataclass
class TracerStackFrame:
    name: str
    locals: Dict[str, Any]
    call_line: int
    frame_type: str = "function"  # function, class, module

class EnhancedUniversalTracer:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the tracer's state for a new execution."""
        self.trace_events: List[TraceEvent] = []
        self.source_lines: List[str] = []
        self.step_counter = 0
        self.call_stack: List[TracerStackFrame] = []
        self.is_tracing = True
        self.captured_output = []
        self.loop_iterations = {}  # Track loop iterations by line number
        self.defined_functions = set()  # Track which functions have been defined
        self.in_function_definition = False
        self.call_sequence_stack = []  # Track the sequence of line calls for recursive functions

    def add_event(self, line_no, event_type, code, note=None, output=None, call_sequence=None):
        """Add a new trace event, capturing the entire execution state."""
        if not self.is_tracing:
            return
        
        # Temporarily disable tracing to prevent recursion
        self.is_tracing = False
        
        self.step_counter += 1
        variables, stack_snapshot = self._capture_state_snapshot()
        
        event = TraceEvent(
            step=self.step_counter,
            line=line_no,
            event=event_type,
            code=code.strip() if code else "",
            variables=variables,
            stack=stack_snapshot,
            note=note,
            output=output,
            call_sequence=call_sequence
        )
        self.trace_events.append(event)
        
        # Re-enable tracing
        self.is_tracing = True

    def _capture_state_snapshot(self):
        """Create a comprehensive snapshot of variables and call stack."""
        if not self.call_stack:
            return {"global": {}, "local": {}}, []

        # Global variables from the module frame
        global_vars = {}
        if self.call_stack:
            module_frame = self.call_stack[0]
            global_vars = {
                k: self._serialize_value(v)
                for k, v in module_frame.locals.items() 
                if not k.startswith('__') and k not in ['print', 'builtins'] and not callable(v)
            }

        # Local variables from the current frame
        local_vars = {}
        if len(self.call_stack) > 1:
            current_frame = self.call_stack[-1]
            local_vars = {
                k: self._serialize_value(v)
                for k, v in current_frame.locals.items()
                if not k.startswith('__') and not callable(v)
            }

        variables_snapshot = {"global": global_vars, "local": local_vars}

        # Stack visualization (skip module frame for clarity)
        stack_snapshot = []
        for frame in self.call_stack[1:]:
            stack_snapshot.append({
                "function": frame.name,
                "type": frame.frame_type,
                "locals": {
                    k: self._serialize_value(v)
                    for k, v in frame.locals.items() 
                    if not k.startswith('__') and not callable(v)
                },
                "line": frame.call_line
            })

        return variables_snapshot, stack_snapshot

    def _serialize_value(self, value):
        """Enhanced serialization for various Python types."""
        if value is None:
            return None
        if isinstance(value, (int, float, str, bool)):
            return value
        if callable(value):
            if hasattr(value, '__name__'):
                return f"<function {value.__name__}>"
            return "<function>"
        
        try:
            if isinstance(value, (list, tuple)):
                serialized = [self._serialize_value(v) for v in value[:10]]  # Limit for readability
                if len(value) > 10:
                    serialized.append(f"... ({len(value) - 10} more items)")
                return serialized
            
            if isinstance(value, dict):
                items = list(value.items())[:10]
                serialized = {str(k): self._serialize_value(v) for k, v in items}
                if len(value) > 10:
                    serialized["..."] = f"({len(value) - 10} more items)"
                return serialized
            
            if isinstance(value, set):
                return list(value)[:10]
            
            # Handle custom objects
            if hasattr(value, '__dict__'):
                return f"<{type(value).__name__} object>"
            
            return str(value)[:200]  # Truncate long strings
        except Exception:
            return f"<unserializable: {type(value).__name__}>"

    def _get_source_line(self, line_no):
        """Safely get a line of source code."""
        if 0 < line_no <= len(self.source_lines):
            return self.source_lines[line_no - 1]
        return ""

    def _analyze_ast_node(self, code_line, line_no, frame_locals=None):
        """Enhanced AST analysis for better event categorization including math operations and comparisons."""
        try:
            # Handle compound statements that might not parse as single expressions
            if any(keyword in code_line for keyword in ['if ', 'elif ', 'else:', 'for ', 'while ', 'def ', 'class ', 'try:', 'except', 'finally:', 'with ', 'import ', 'from ']):
                return self._handle_compound_statement(code_line, line_no)
            
            # Try to parse as expression or simple statement
            try:
                node = ast.parse(code_line.strip()).body[0]
            except (SyntaxError, IndexError):
                # If parsing fails, try to detect operations by text analysis
                return self._detect_operations(code_line)
            
            return self._categorize_ast_node(node, code_line, frame_locals)
            
        except Exception:
            return self._detect_operations(code_line)

    def _detect_operations(self, code_line):
        """Detect math operations and comparisons in code lines that might not parse properly."""
        stripped = code_line.strip()
        
        # Check for comparison operations first
        comparison_ops = ['==', '!=', '<=', '>=', '<', '>']
        for op in comparison_ops:
            if op in stripped:
                return EventType.COMPARISON, f"Comparison operation in: {stripped}"
        
        # Check for multiplication
        if '*' in stripped and not '**' in stripped:
            return EventType.MULTIPLICATION, f"Multiplication operation in: {stripped}"
        
        # Check for addition
        if '+' in stripped and not '+=' in stripped:
            return EventType.ADDITION, f"Addition operation in: {stripped}"
        
        # Check for subtraction
        if '-' in stripped and not '-=' in stripped and not '--' in stripped:
            return EventType.SUBTRACTION, f"Subtraction operation in: {stripped}"
        
        # Check for division
        if '/' in stripped and not '//' in stripped and not '/=' in stripped:
            return EventType.DIVISION, f"Division operation in: {stripped}"
        
        return EventType.EXPRESSION, "Evaluating expression"

    def _analyze_expression_for_operations(self, node, code_line, frame_locals=None):
        """Analyze expressions to detect specific operations with actual values."""
        if isinstance(node, ast.Compare):
            # Handle comparison operations
            return self._analyze_comparison(node, code_line, frame_locals)
        elif isinstance(node, ast.BinOp):
            # Handle binary operations (math)
            return self._analyze_math_operation(node, code_line, frame_locals)
        
        return EventType.EXPRESSION, "Evaluating expression"

    def _analyze_comparison(self, node, code_line, frame_locals):
        """Analyze comparison operations and show detailed results."""
        try:
            left_val = self._get_value_from_node(node.left, frame_locals)
            
            if len(node.ops) == 1 and len(node.comparators) == 1:
                # Single comparison
                op = node.ops[0]
                right_val = self._get_value_from_node(node.comparators[0], frame_locals)
                
                op_symbols = {
                    ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=',
                    ast.Gt: '>', ast.GtE: '>=', ast.Is: 'is', ast.IsNot: 'is not',
                    ast.In: 'in', ast.NotIn: 'not in'
                }
                
                op_str = op_symbols.get(type(op), '?')
                
                if left_val is not None and right_val is not None:
                    # Evaluate the comparison
                    try:
                        if isinstance(op, ast.Eq):
                            result = left_val == right_val
                        elif isinstance(op, ast.NotEq):
                            result = left_val != right_val
                        elif isinstance(op, ast.Lt):
                            result = left_val < right_val
                        elif isinstance(op, ast.LtE):
                            result = left_val <= right_val
                        elif isinstance(op, ast.Gt):
                            result = left_val > right_val
                        elif isinstance(op, ast.GtE):
                            result = left_val >= right_val
                        elif isinstance(op, ast.Is):
                            result = left_val is right_val
                        elif isinstance(op, ast.IsNot):
                            result = left_val is not right_val
                        elif isinstance(op, ast.In):
                            result = left_val in right_val
                        elif isinstance(op, ast.NotIn):
                            result = left_val not in right_val
                        else:
                            result = "Unknown"
                        
                        # Create detailed explanation
                        explanation = f"Comparing {left_val} {op_str} {right_val} → {result}"
                        
                        # Add reasoning for the result
                        if result is True:
                            if isinstance(op, ast.Eq):
                                explanation += f" (True because {left_val} equals {right_val})"
                            elif isinstance(op, ast.NotEq):
                                explanation += f" (True because {left_val} is not equal to {right_val})"
                            elif isinstance(op, ast.Lt):
                                explanation += f" (True because {left_val} is less than {right_val})"
                            elif isinstance(op, ast.LtE):
                                explanation += f" (True because {left_val} is less than or equal to {right_val})"
                            elif isinstance(op, ast.Gt):
                                explanation += f" (True because {left_val} is greater than {right_val})"
                            elif isinstance(op, ast.GtE):
                                explanation += f" (True because {left_val} is greater than or equal to {right_val})"
                        elif result is False:
                            if isinstance(op, ast.Eq):
                                explanation += f" (False because {left_val} does not equal {right_val})"
                            elif isinstance(op, ast.NotEq):
                                explanation += f" (False because {left_val} equals {right_val})"
                            elif isinstance(op, ast.Lt):
                                explanation += f" (False because {left_val} is not less than {right_val})"
                            elif isinstance(op, ast.LtE):
                                explanation += f" (False because {left_val} is greater than {right_val})"
                            elif isinstance(op, ast.Gt):
                                explanation += f" (False because {left_val} is not greater than {right_val})"
                            elif isinstance(op, ast.GtE):
                                explanation += f" (False because {left_val} is less than {right_val})"
                        
                        return EventType.COMPARISON, explanation
                        
                    except Exception as e:
                        return EventType.COMPARISON, f"Comparison: {left_val} {op_str} {right_val} (evaluation error: {e})"
                
                return EventType.COMPARISON, f"Comparison: {op_str} operation"
            else:
                # Multiple comparisons (chained)
                return EventType.COMPARISON, f"Chained comparison: {code_line.strip()}"
                
        except Exception:
            return EventType.COMPARISON, f"Comparison operation: {code_line.strip()}"

    def _analyze_math_operation(self, node, code_line, frame_locals=None):
        """Analyze math operations with actual values."""
        # Try to get the actual values being operated on
        left_val = self._get_value_from_node(node.left, frame_locals)
        right_val = self._get_value_from_node(node.right, frame_locals)
        
        if isinstance(node.op, ast.Mult):
            if left_val is not None and right_val is not None:
                result = left_val * right_val
                return EventType.MULTIPLICATION, f"Multiplication: {left_val} * {right_val} = {result}"
            return EventType.MULTIPLICATION, f"Multiplication: {code_line.strip()}"
        elif isinstance(node.op, ast.Add):
            if left_val is not None and right_val is not None:
                result = left_val + right_val
                return EventType.ADDITION, f"Addition: {left_val} + {right_val} = {result}"
            return EventType.ADDITION, f"Addition: {code_line.strip()}"
        elif isinstance(node.op, ast.Sub):
            if left_val is not None and right_val is not None:
                result = left_val - right_val
                return EventType.SUBTRACTION, f"Subtraction: {left_val} - {right_val} = {result}"
            return EventType.SUBTRACTION, f"Subtraction: {code_line.strip()}"
        elif isinstance(node.op, ast.Div):
            if left_val is not None and right_val is not None:
                result = left_val / right_val
                return EventType.DIVISION, f"Division: {left_val} / {right_val} = {result}"
            return EventType.DIVISION, f"Division: {code_line.strip()}"
        elif isinstance(node.op, ast.FloorDiv):
            if left_val is not None and right_val is not None:
                result = left_val // right_val
                return EventType.DIVISION, f"Floor division: {left_val} // {right_val} = {result}"
            return EventType.DIVISION, f"Floor division: {code_line.strip()}"
        
        return EventType.EXPRESSION, "Evaluating expression"

    def _handle_compound_statement(self, code_line, line_no):
        """Handle compound statements that need special treatment."""
        stripped = code_line.strip()
        
        if stripped.startswith('if ') or stripped.startswith('elif '):
            return EventType.IF, "Evaluating conditional statement"
        elif stripped == 'else:':
            return EventType.IF, "Executing else block"
        elif stripped.startswith('for '):
            # Track loop iterations
            if line_no not in self.loop_iterations:
                self.loop_iterations[line_no] = 0
            self.loop_iterations[line_no] += 1
            return EventType.LOOP, f"Loop iteration {self.loop_iterations[line_no]}"
        elif stripped.startswith('while '):
            if line_no not in self.loop_iterations:
                self.loop_iterations[line_no] = 0
            self.loop_iterations[line_no] += 1
            return EventType.LOOP, f"While loop iteration {self.loop_iterations[line_no]}"
        elif stripped.startswith('def '):
            func_name = stripped.split('(')[0].replace('def ', '')
            return EventType.DEF, f"Defining function '{func_name}'"
        elif stripped.startswith('class '):
            class_name = stripped.split('(')[0].split(':')[0].replace('class ', '')
            return EventType.CLASS, f"Defining class '{class_name}'"
        elif stripped.startswith(('import ', 'from ')):
            return EventType.IMPORT, f"Importing: {stripped}"
        elif stripped.startswith('with '):
            return EventType.WITH, "Entering 'with' block"
        elif stripped == 'try:':
            return EventType.TRY, "Entering try block"
        elif stripped.startswith('except'):
            return EventType.EXCEPT, "Handling exception"
        elif stripped == 'finally:':
            return EventType.FINALLY, "Executing finally block"
        
        return EventType.EXPRESSION, "Evaluating expression"

    def _categorize_ast_node(self, node, code_line, frame_locals=None):
        """Categorize AST nodes into appropriate event types."""
        if isinstance(node, ast.Assign):
            targets = []
            for target in node.targets:
                if hasattr(target, 'id'):
                    targets.append(target.id)
                elif hasattr(target, 'elts'):  # Multiple assignment
                    targets.extend([elt.id for elt in target.elts if hasattr(elt, 'id')])
                else:
                    targets.append('[complex target]')
            
            target_str = ', '.join(targets)
            
            # Check if the assignment involves operations
            if isinstance(node.value, (ast.BinOp, ast.Compare)):
                return self._analyze_expression_for_operations(node.value, code_line, frame_locals)
            
            return EventType.ASSIGN, f"Assigning value to '{target_str}'"
        
        elif isinstance(node, ast.AugAssign):  # += -= *= etc
            target = node.target.id if hasattr(node.target, 'id') else '[complex target]'
            op_map = {
                ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/=',
                ast.Mod: '%=', ast.Pow: '**=', ast.FloorDiv: '//='
            }
            op_str = op_map.get(type(node.op), 'op=')
            return EventType.ASSIGN, f"Augmented assignment: {target} {op_str}"
        
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                func_name = self._extract_function_name(node.value)
                if func_name == 'print':
                    return EventType.PRINT, "Print statement"
                return EventType.EXPRESSION, f"Function call: {func_name}()"
            elif isinstance(node.value, (ast.BinOp, ast.Compare)):
                return self._analyze_expression_for_operations(node.value, code_line, frame_locals)
            return EventType.EXPRESSION, "Evaluating expression"
        
        elif isinstance(node, ast.Return):
            if isinstance(node.value, (ast.BinOp, ast.Compare)):
                return self._analyze_expression_for_operations(node.value, code_line, frame_locals)
            return EventType.RETURN, "Return statement"
        
        return EventType.EXPRESSION, "Evaluating expression"

    def _get_value_from_node(self, node, frame_locals):
        """Extract actual value from an AST node using current frame locals."""
        if frame_locals is None:
            return None
            
        try:
            if isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            elif isinstance(node, ast.Num):  # Older Python versions
                return node.n
            elif isinstance(node, ast.Str):  # Older Python versions
                return node.s
            elif isinstance(node, ast.Name):
                # Variable reference
                if node.id in frame_locals:
                    return frame_locals[node.id]
            elif isinstance(node, ast.Call):
                # Function call - we can't easily evaluate this without executing
                func_name = self._extract_function_name(node)
                return f"{func_name}(...)"
            elif isinstance(node, ast.BinOp):
                # Nested binary operation - try to evaluate recursively
                left_val = self._get_value_from_node(node.left, frame_locals)
                right_val = self._get_value_from_node(node.right, frame_locals)
                
                if left_val is not None and right_val is not None:
                    if isinstance(node.op, ast.Add):
                        return left_val + right_val
                    elif isinstance(node.op, ast.Sub):
                        return left_val - right_val
                    elif isinstance(node.op, ast.Mult):
                        return left_val * right_val
                    elif isinstance(node.op, ast.Div):
                        return left_val / right_val
                    elif isinstance(node.op, ast.FloorDiv):
                        return left_val // right_val
                    elif isinstance(node.op, ast.Mod):
                        return left_val % right_val
                    elif isinstance(node.op, ast.Pow):
                        return left_val ** right_val
                        
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            # If we can't evaluate, return None
            pass
            
        return None

    def _extract_function_name(self, call_node):
        """Extract function name from call node."""
        if hasattr(call_node.func, 'id'):
            return call_node.func.id
        elif hasattr(call_node.func, 'attr'):
            return call_node.func.attr
        return "unknown"

    def _get_call_sequence(self):
        """Generate call sequence string showing the line path."""
        if len(self.call_sequence_stack) > 1:
            return " -> ".join([f"line {line}" for line in self.call_sequence_stack])
        return None

    def trace_execution(self, code):
        """Main method to trace Python code execution."""
        self.reset()
        self.source_lines = code.strip().split('\n')
        self.add_event(0, EventType.START, "", note="Execution begins")

        # Enhanced print function that captures output
        def traced_print(*args, **kwargs):
            output = ' '.join(str(arg) for arg in args)
            self.captured_output.append(output)
            
            # Don't add print event here - it will be handled by the line tracer
            # Call original print
            return builtins.print(*args, **kwargs)

        # Create execution namespace
        exec_namespace = {
            'print': traced_print,
            '__builtins__': builtins,
        }

        try:
            # Execute with tracing
            sys.settrace(self._trace_function)
            exec(code, exec_namespace, exec_namespace)
            
        except Exception as e:
            # Enhanced exception handling
            tb = sys.exc_info()[2]
            line_no = tb.tb_lineno if tb else 0
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            self.add_event(
                line_no, 
                EventType.EXCEPTION, 
                self._get_source_line(line_no), 
                note=f"Exception occurred: {error_msg}"
            )
            
        finally:
            sys.settrace(None)

        return self._prepare_final_output()

    def _trace_function(self, frame, event, arg):
        """Enhanced trace function with better event handling."""
        if not self.is_tracing or frame.f_code.co_filename != '<string>':
            return self._trace_function

        func_name = frame.f_code.co_name
        line_no = frame.f_lineno

        if event == 'call':
            # Handle module-level execution
            if func_name == '<module>':
                if not self.call_stack:
                    self.call_stack.append(TracerStackFrame(
                        name="<module>", 
                        locals=frame.f_globals.copy(), 
                        call_line=0,
                        frame_type="module"
                    ))
                return self._trace_function

            # Handle function calls
            caller_frame = frame.f_back
            call_site_line = caller_frame.f_lineno if caller_frame else 0
            
            # Update call sequence stack
            self.call_sequence_stack.append(call_site_line)
            
            # Get function arguments
            arg_names = frame.f_code.co_varnames[:frame.f_code.co_argcount]
            args_info = []
            for name in arg_names:
                if name in frame.f_locals:
                    value = self._serialize_value(frame.f_locals[name])
                    args_info.append(f"{name}={value}")
            
            args_str = ", ".join(args_info)
            note = f"Calling function '{func_name}({args_str})'"
            
            call_line_code = self._get_source_line(call_site_line)
            call_sequence = self._get_call_sequence()
            
            self.add_event(call_site_line, EventType.CALL, call_line_code, note=note, call_sequence=call_sequence)
            
            # Add to call stack
            self.call_stack.append(TracerStackFrame(
                name=func_name, 
                locals=frame.f_locals.copy(), 
                call_line=call_site_line,
                frame_type="function"
            ))
            
            # Add ENTER event for function definition line
            func_def_line = frame.f_code.co_firstlineno
            func_def_code = self._get_source_line(func_def_line)
            enter_note = f"Entering function '{func_name}'"
            
            self.add_event(func_def_line, EventType.ENTER, func_def_code, note=enter_note, call_sequence=call_sequence)

        elif event == 'return':
            if self.call_stack and func_name != '<module>':
                return_value = self._serialize_value(arg)
                note = f"Returning from '{func_name}' with value: {return_value}"
                
                # Find the return statement line
                return_line_code = self._get_source_line(line_no)
                call_sequence = self._get_call_sequence()
                
                self.add_event(line_no, EventType.RETURN, return_line_code, note=note, call_sequence=call_sequence)
                
                self.call_stack.pop()
                if self.call_sequence_stack:
                    self.call_sequence_stack.pop()

        elif event == 'line':
            # Update current frame's locals
            if self.call_stack:
                self.call_stack[-1].locals.update(frame.f_locals)
            
            code_line = self._get_source_line(line_no)
            if not code_line.strip():
                return self._trace_function

            # Skip function definition lines that we've already processed
            if code_line.strip().startswith('def '):
                func_name_from_line = code_line.strip().split('(')[0].replace('def ', '').strip()
                if func_name_from_line not in self.defined_functions:
                    self.defined_functions.add(func_name_from_line)
                    event_type, note = self._analyze_ast_node(code_line, line_no, frame.f_locals)
                    call_sequence = self._get_call_sequence()
                    self.add_event(line_no, event_type, code_line, note=note, call_sequence=call_sequence)
                return self._trace_function

            # Analyze and categorize the line with current frame locals
            event_type, note = self._analyze_ast_node(code_line, line_no, frame.f_locals)
            call_sequence = self._get_call_sequence()
            
            # Handle print statements specially
            if 'print(' in code_line and event_type in [EventType.EXPRESSION, EventType.PRINT]:
                # Extract what's being printed for the note
                try:
                    # Try to get the actual output from captured_output
                    if self.captured_output:
                        last_output = self.captured_output[-1]
                        note = f"Print: {last_output}"
                    else:
                        note = "Print statement"
                    event_type = EventType.PRINT
                except:
                    note = "Print statement"
                    event_type = EventType.PRINT
            
            self.add_event(line_no, event_type, code_line, note=note, call_sequence=call_sequence)

        return self._trace_function

    def _prepare_final_output(self):
        """Convert internal events to final JSON output."""
        final_events = []
        
        for event in self.trace_events:
            event_dict = asdict(event)
            event_dict['event'] = event.event.value
            
            # Remove None/empty optional fields
            if not event_dict.get('note'):
                del event_dict['note']
            if event_dict.get('output') is None:
                del event_dict['output']
            if not event_dict.get('call_sequence'):
                del event_dict['call_sequence']
            
            final_events.append(event_dict)
        
        return json.dumps(final_events, indent=2)

def trace_python_code(code_string, output_format="json"):
    """
    Main entry point for tracing Python code.
    
    Args:
        code_string (str): The Python code to trace
        output_format (str): Output format (currently only "json" supported)
    
    Returns:
        str: JSON string containing the execution trace
    """
    tracer = EnhancedUniversalTracer()
    return tracer.trace_execution(code_string)

# Test the tracer with examples including comparisons
if __name__ == "__main__":
    # Test with factorial
    test_code_factorial = """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
print(factorial(5))"""
    
    # Test with comparison operations
    test_code_comparison = """x = 5
y = 10
z = 5
print("Testing comparisons:")
result1 = x == z
print(f"{x} == {z} is {result1}")
result2 = x < y
print(f"{x} < {y} is {result2}")
result3 = y >= x
print(f"{y} >= {x} is {result3}")
if x == z:
    print("x equals z")
else:
    print("x does not equal z")"""
    
    # Test with mixed operations
    test_code_mixed = """def compare_and_calculate(a, b):
    if a > b:
        return a * 2
    elif a == b:
        return a + b
    else:
        return b - a

result = compare_and_calculate(3, 7)
print(f"Result: {result}")"""
    
    print("=== FACTORIAL EXAMPLE ===")
    trace_output = trace_python_code(test_code_factorial)
    print(trace_output)
    
    print("\n=== COMPARISON EXAMPLE ===")
    trace_output = trace_python_code(test_code_comparison)
    print(trace_output)
    
    print("\n=== MIXED OPERATIONS EXAMPLE ===")
    trace_output = trace_python_code(test_code_mixed)
    print(trace_output)