import sys
import ast
import json
import traceback
from typing import Any, Dict, List, Optional
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
    COMPARISON = "COMPARISON"

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
    call_sequence: Optional[str] = None

@dataclass
class TracerStackFrame:
    name: str
    locals: Dict[str, Any]
    globals: Dict[str, Any]  # Store globals for proper eval context
    call_line: int
    frame_type: str = "function"

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
        self.current_print_output = None # Accurately captures print output per line
        self.loop_iterations = {}
        self.defined_functions = set()
        self.call_sequence_stack = []

    def add_event(self, line_no, event_type, code, note=None, output=None, call_sequence=None):
        """Add a new trace event, capturing the entire execution state."""
        if not self.is_tracing: return
        
        self.is_tracing = False
        self.step_counter += 1
        variables, stack_snapshot = self._capture_state_snapshot()
        
        event = TraceEvent(
            step=self.step_counter, line=line_no, event=event_type,
            code=code.strip() if code else "", variables=variables,
            stack=stack_snapshot, note=note, output=output, call_sequence=call_sequence
        )
        self.trace_events.append(event)
        self.is_tracing = True

    def _capture_state_snapshot(self):
        """Create a comprehensive snapshot of variables and call stack."""
        if not self.call_stack: return {"global": {}, "local": {}}, []

        module_frame = self.call_stack[0]
        global_vars = {
            k: self._serialize_value(v)
            for k, v in module_frame.locals.items() 
            if not k.startswith('__') and k not in ['print', 'builtins'] and not callable(v)
        }

        current_frame = self.call_stack[-1]
        local_vars = {
            k: self._serialize_value(v) for k, v in current_frame.locals.items()
            if not k.startswith('__') and not callable(v) and k not in global_vars
        }

        variables_snapshot = {"global": global_vars, "local": local_vars}
        stack_snapshot = [
            {
                "function": frame.name, "type": frame.frame_type, "line": frame.call_line,
                "locals": {
                    k: self._serialize_value(v) for k, v in frame.locals.items() 
                    if not k.startswith('__') and not callable(v) and k not in frame.globals
                }
            } for frame in self.call_stack[1:]
        ]
        return variables_snapshot, stack_snapshot

    def _serialize_value(self, value):
        """Enhanced serialization for various Python types."""
        if isinstance(value, (int, float, str, bool, type(None))): return value
        if callable(value): return f"<function {getattr(value, '__name__', 'unknown')}>"
        try:
            if isinstance(value, (list, tuple)): return [self._serialize_value(v) for v in value]
            if isinstance(value, dict): return {str(k): self._serialize_value(v) for k, v in value.items()}
            if isinstance(value, set): return sorted([self._serialize_value(v) for v in value])
            if hasattr(value, '__dict__'): return f"<{type(value).__name__} object>"
            return str(value)[:200]
        except Exception: return f"<unserializable: {type(value).__name__}>"

    def _get_source_line(self, line_no):
        return self.source_lines[line_no - 1] if 0 < line_no <= len(self.source_lines) else ""

    def _get_value_from_node(self, node, frame):
        """
        CORRECTED CORE EVALUATION: Robustly get a value by compiling and evaluating the node.
        This is the core fix that replaces the brittle, manual evaluation.
        """
        try:
            expr = ast.Expression(body=node)
            code_obj = compile(expr, filename='<ast>', mode='eval')
            return eval(code_obj, frame.f_globals, frame.f_locals)
        except Exception:
            return None # Cannot evaluate (e.g., it's a call we shouldn't execute here)

    def _analyze_comparison(self, node, frame):
        """Analyze a comparison operation using the robust evaluation method."""
        left_val = self._get_value_from_node(node.left, frame)
        op_map = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in', ast.NotIn: 'not in'}
        op_str = op_map.get(type(node.ops[0]), '?')
        right_val = self._get_value_from_node(node.comparators[0], frame)

        if left_val is not None and right_val is not None:
            result = eval(f"a {op_str} b", {"a": left_val, "b": right_val})
            return EventType.COMPARISON, f"Comparing {self._serialize_value(left_val)} {op_str} {self._serialize_value(right_val)} → {result}"
        return EventType.COMPARISON, "Evaluating comparison"

    def _analyze_math_operation(self, node, frame):
        """Analyze a math operation using the robust evaluation method."""
        op_map = {ast.Add: ('+', EventType.ADDITION), ast.Sub: ('-', EventType.SUBTRACTION), ast.Mult: ('*', EventType.MULTIPLICATION), ast.Div: ('/', EventType.DIVISION), ast.FloorDiv: ('//', EventType.DIVISION)}
        op_str, event_type = op_map.get(type(node.op), ('?', EventType.EXPRESSION))

        left_val = self._get_value_from_node(node.left, frame)
        right_val = self._get_value_from_node(node.right, frame)

        if left_val is not None and right_val is not None:
            try:
                result = eval(f"a {op_str} b", {"a": left_val, "b": right_val})
                return event_type, f"Calculation: {self._serialize_value(left_val)} {op_str} {self._serialize_value(right_val)} = {self._serialize_value(result)}"
            except Exception:
                pass # Fallback on evaluation error
        return event_type, f"Executing {event_type.name.lower()}"

    def _extract_function_name(self, call_node):
        if isinstance(call_node.func, ast.Name): return call_node.func.id
        if isinstance(call_node.func, ast.Attribute): return call_node.func.attr
        return "unknown"

    def _get_call_sequence(self):
        return " -> ".join(f"line {line}" for line in self.call_sequence_stack) if len(self.call_sequence_stack) > 1 else None

    def trace_execution(self, code):
        """Main method to trace Python code execution."""
        self.reset()
        self.source_lines = code.strip().split('\n')
        self.add_event(0, EventType.START, "", note="Execution begins")

        def traced_print(*args, **kwargs):
            self.current_print_output = ' '.join(str(arg) for arg in args)
            return builtins.print(*args, **kwargs)

        exec_namespace = {'print': traced_print, '__builtins__': builtins}

        try:
            sys.settrace(self._trace_function)
            exec(code, exec_namespace, exec_namespace)
        except Exception as e:
            tb = sys.exc_info()[2]
            line_no = tb.tb_lineno if tb else 0
            self.add_event(line_no, EventType.EXCEPTION, self._get_source_line(line_no), note=f"Exception: {type(e).__name__}: {e}")
        finally:
            sys.settrace(None)
        
        return self._prepare_final_output()

    def _trace_function(self, frame, event, arg):
        """The core trace function passed to sys.settrace."""
        if not self.is_tracing or frame.f_code.co_filename != '<string>':
            return self._trace_function

        line_no = frame.f_lineno
        
        if event == 'call':
            func_name = frame.f_code.co_name
            if func_name == '<module>':
                if not self.call_stack: self.call_stack.append(TracerStackFrame("<module>", frame.f_locals, frame.f_globals, 0, "module"))
                return self._trace_function
            
            caller_frame = frame.f_back
            call_site_line = caller_frame.f_lineno if caller_frame else 0
            self.call_sequence_stack.append(call_site_line)
            
            arg_names = frame.f_code.co_varnames[:frame.f_code.co_argcount]
            args_info = [f"{name}={self._serialize_value(frame.f_locals.get(name))}" for name in arg_names]
            
            self.add_event(call_site_line, EventType.CALL, self._get_source_line(call_site_line), note=f"Calling function '{func_name}({', '.join(args_info)})'", call_sequence=self._get_call_sequence())
            self.call_stack.append(TracerStackFrame(func_name, frame.f_locals, frame.f_globals, call_site_line))
            self.add_event(frame.f_code.co_firstlineno, EventType.ENTER, self._get_source_line(frame.f_code.co_firstlineno), note=f"Entering function '{func_name}'", call_sequence=self._get_call_sequence())

        elif event == 'return':
            if self.call_stack and self.call_stack[-1].name != '<module>':
                func_name = self.call_stack[-1].name
                self.add_event(line_no, EventType.RETURN, self._get_source_line(line_no), note=f"Returning from '{func_name}' with value: {self._serialize_value(arg)}", call_sequence=self._get_call_sequence())
                self.call_stack.pop()
                if self.call_sequence_stack: self.call_sequence_stack.pop()

        elif event == 'line':
            if self.call_stack: self.call_stack[-1].locals.update(frame.f_locals)
            
            code_line = self._get_source_line(line_no)
            if not code_line.strip(): return self._trace_function

            event_type, note, output = EventType.EXPRESSION, "Executing statement", None

            try:
                node = ast.parse(code_line.strip()).body[0]
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    if isinstance(node, (ast.If)):
                        event_type, note = self._analyze_comparison(node.test, frame)
                    else: # For, While
                        self.loop_iterations[line_no] = self.loop_iterations.get(line_no, 0) + 1
                        event_type, note = EventType.LOOP, f"Loop iteration {self.loop_iterations[line_no]}"
                elif isinstance(node, ast.Assign):
                    target_str = ', '.join(t.id for t in node.targets if isinstance(t, ast.Name))
                    event_type, note = self._analyze_math_operation(node.value, frame) if isinstance(node.value, ast.BinOp) else (EventType.ASSIGN, f"Assigning to '{target_str}'")
                elif isinstance(node, ast.AugAssign):
                    target_str = node.target.id if isinstance(node.target, ast.Name) else 'variable'
                    event_type, note = EventType.ASSIGN, f"Updating '{target_str}'"
                elif isinstance(node, ast.Return):
                     event_type, note = self._analyze_math_operation(node.value, frame) if isinstance(node.value, ast.BinOp) else (EventType.RETURN, "Returning from function")
                elif isinstance(node, ast.Expr):
                    value_node = node.value
                    if isinstance(value_node, ast.Call) and self._extract_function_name(value_node) == 'print':
                        event_type, output = EventType.PRINT, self.current_print_output
                        note = f"Printing: {output}"
                        self.current_print_output = None # Consume the output
                    elif isinstance(value_node, ast.BinOp):
                        event_type, note = self._analyze_math_operation(value_node, frame)
                    elif isinstance(value_node, ast.Compare):
                        event_type, note = self._analyze_comparison(value_node, frame)
            except (SyntaxError, IndexError):
                pass # Use default event/note for lines that don't parse to a simple statement

            self.add_event(line_no, event_type, code_line, note=note, output=output, call_sequence=self._get_call_sequence())

        return self._trace_function

    def _prepare_final_output(self):
        """Convert internal events to the final JSON output string."""
        final_events = []
        for event in self.trace_events:
            event_dict = asdict(event)
            event_dict['event'] = event.event.value
            final_events.append({k: v for k, v in event_dict.items() if v is not None and v != [] and v!={}})
        return json.dumps(final_events, indent=2)

def trace_python_code(code_string: str, output_format="json") -> str:
    """
    Main entry point for tracing Python code.
    
    Args:
        code_string (str): The Python code to trace.
        output_format (str): The desired output format (currently only "json" is supported).
    
    Returns:
        str: A JSON string representing the execution trace.
    """
    if output_format != "json":
        return json.dumps({"error": "Unsupported output format specified."})
    tracer = EnhancedUniversalTracer()
    return tracer.trace_execution(code_string)

# Test with a comprehensive example
if __name__ == "__main__":
    test_code_mixed = """
def check_value(val):
    print(f"Checking {val}")
    if val > 100:
        res = val - 100
        print("Over 100")
        return res
    elif val == 100:
        return 0
    else:
        res = 100 - val
        return res

x = 50
y = 105
z = check_value(x + y)
print(f"Final result is {z}")
"""
    
    print("\n=== COMPREHENSIVE EXAMPLE ===")
    trace_output = trace_python_code(test_code_mixed)
    print(trace_output)