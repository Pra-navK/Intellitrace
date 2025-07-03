import sys
import ast
import json
import traceback
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import builtins
import random

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
    INPUT = "INPUT"

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
    globals: Dict[str, Any]
    call_line: int
    frame_type: str = "function"

class EnhancedUniversalTracer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trace_events: List[TraceEvent] = []
        self.source_lines: List[str] = []
        self.step_counter = 0
        self.call_stack: List[TracerStackFrame] = []
        self.is_tracing = True
        self.last_global_print_output = None
        self.loop_iterations = {}
        self.defined_functions = set()
        self.call_sequence_stack = []

    def add_event(self, line_no, event_type, code, note=None, output=None, call_sequence=None):
        if not self.is_tracing: return
        # A print event updates the last known output, keeping its type
        if event_type == EventType.PRINT:
             self.last_global_print_output = output

        self.is_tracing = False
        self.step_counter += 1
        variables, stack_snapshot = self._capture_state_snapshot()
        event = TraceEvent(step=self.step_counter, line=line_no, event=event_type, code=code.strip() if code else "", variables=variables, stack=stack_snapshot, note=note, output=output, call_sequence=call_sequence)
        self.trace_events.append(event)
        self.is_tracing = True

    def _capture_state_snapshot(self):
        if not self.call_stack: return {}, []
        
        module_frame = self.call_stack[0]
        global_vars = {
            k: self._serialize_value(v)
            for k, v in module_frame.locals.items()
            if not k.startswith('__') and k not in ['print', 'input', 'builtins'] and not callable(v)
        }

        # Add the persistent print output directly to the global_vars dictionary
        if self.last_global_print_output is not None:
            # We serialize it here to ensure it's JSON-safe
            global_vars["print_output"] = self._serialize_value(self.last_global_print_output)

        current_frame = self.call_stack[-1]
        local_vars = {}
        if current_frame.name != "<module>":
             local_vars = {
                k: self._serialize_value(v)
                for k, v in current_frame.locals.items()
                if not k.startswith('__') and not callable(v) and k not in global_vars
             }

        variables_snapshot = { "global": global_vars, "local": local_vars }
        stack_snapshot = [
            {
                "function": frame.name, "type": frame.frame_type, "line": frame.call_line,
                "locals": { k: self._serialize_value(v) for k, v in frame.locals.items() if not k.startswith('__') and not callable(v) and k not in frame.globals }
            } for frame in self.call_stack[1:]
        ]
        
        return variables_snapshot, stack_snapshot

    def _serialize_value(self, value):
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
        try:
            expr = ast.Expression(body=node)
            code_obj = compile(expr, filename='<ast>', mode='eval')
            return eval(code_obj, frame.f_globals, frame.f_locals)
        except Exception: return None

    def _analyze_comparison(self, node, frame):
        left_val = self._get_value_from_node(node.left, frame)
        op_map = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in', ast.NotIn: 'not in'}
        op_str = op_map.get(type(node.ops[0]), '?')
        right_val = self._get_value_from_node(node.comparators[0], frame)
        if left_val is not None and right_val is not None:
            try:
                result = eval(f"a {op_str} b", {"a": left_val, "b": right_val})
                return EventType.COMPARISON, f"Comparing {self._serialize_value(left_val)} {op_str} {self._serialize_value(right_val)} → {result}"
            except: pass
        return EventType.COMPARISON, "Evaluating comparison"

    def _analyze_math_operation(self, node, frame):
        op_map = {ast.Add: ('+', EventType.ADDITION), ast.Sub: ('-', EventType.SUBTRACTION), ast.Mult: ('*', EventType.MULTIPLICATION), ast.Div: ('/', EventType.DIVISION), ast.FloorDiv: ('//', EventType.DIVISION)}
        op_str, event_type = op_map.get(type(node.op), ('?', EventType.EXPRESSION))
        left_val = self._get_value_from_node(node.left, frame)
        right_val = self._get_value_from_node(node.right, frame)
        if left_val is not None and right_val is not None:
            try:
                result = eval(f"a {op_str} b", {"a": left_val, "b": right_val})
                return event_type, f"Calculation: {self._serialize_value(left_val)} {op_str} {self._serialize_value(right_val)} = {self._serialize_value(result)}"
            except Exception: pass
        return event_type, f"Executing {event_type.name.lower()}"

    def _extract_function_name(self, call_node):
        if isinstance(call_node.func, ast.Name): return call_node.func.id
        if isinstance(call_node.func, ast.Attribute): return call_node.func.attr
        return "unknown"

    def _get_call_sequence(self):
        return " -> ".join(f"line {line}" for line in self.call_sequence_stack) if len(self.call_sequence_stack) > 1 else None

    def trace_execution(self, code):
        self.reset()
        self.source_lines = code.strip().split('\n')
        self.add_event(0, EventType.START, "", note="Execution begins")
        def traced_print(*args, **kwargs): pass
        def mock_input(prompt=""): return str(random.randint(1, 8))
        exec_namespace = { 'print': traced_print, 'input': mock_input, '__builtins__': builtins, '__name__': '__main__' }

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
        if not self.is_tracing or frame.f_code.co_filename != '<string>': return self._trace_function
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
                if isinstance(node, (ast.If)): event_type, note = self._analyze_comparison(node.test, frame)
                elif isinstance(node, (ast.For, ast.While)):
                    self.loop_iterations[line_no] = self.loop_iterations.get(line_no, 0) + 1
                    event_type, note = EventType.LOOP, f"Loop iteration {self.loop_iterations[line_no]}"
                elif isinstance(node, ast.Assign):
                    target_str = ', '.join(t.id for t in node.targets if isinstance(t, ast.Name))
                    event_type, note = self._analyze_math_operation(node.value, frame) if isinstance(node.value, ast.BinOp) else (EventType.ASSIGN, f"Assigning to '{target_str}'")
                elif isinstance(node, ast.AugAssign):
                    target_str = node.target.id if isinstance(node.target, ast.Name) else 'variable'
                    event_type, note = self._analyze_math_operation(node, frame) if isinstance(node.value, ast.BinOp) else (EventType.ASSIGN, f"Updating '{target_str}'")
                elif isinstance(node, ast.Return):
                    event_type, note = self._analyze_math_operation(node.value, frame) if isinstance(node.value, ast.BinOp) else (EventType.RETURN, "Preparing to return")
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call_node = node.value
                    func_name = self._extract_function_name(call_node)
                    if func_name == 'print':
                        event_type = EventType.PRINT
                        try:
                            # *** THIS IS THE NEW TYPE-PRESERVING LOGIC ***
                            arg_values = [self._get_value_from_node(arg, frame) for arg in call_node.args]
                            if len(arg_values) == 1:
                                output = arg_values[0] # Keep original type for single argument
                            else:
                                output = ' '.join(map(str, arg_values)) # Join multiple args as a string
                            note = f"Printing: {str(output)}" # The note is always a string
                        except Exception:
                            output, note = "...", "Printing value(s)"
                    elif func_name == 'input':
                        event_type = EventType.INPUT
                        prompt = self._get_value_from_node(call_node.args[0], frame) if call_node.args else ""
                        note = f"Reading input with prompt: '{prompt}'. Simulated."
            except (SyntaxError, IndexError): pass
            self.add_event(line_no, event_type, code_line, note=note, output=output, call_sequence=self._get_call_sequence())
        return self._trace_function

    def _prepare_final_output(self):
        final_events = []
        for event in self.trace_events:
            event_dict = asdict(event)
            event_dict['event'] = event.event.value
            final_events.append({k: v for k, v in event_dict.items() if v is not None and v != []})
        return json.dumps(final_events, indent=2)

def trace_python_code(code_string: str, output_format="json") -> str:
    if output_format != "json":
        return json.dumps({"error": "Unsupported output format specified."})
    tracer = EnhancedUniversalTracer()
    return tracer.trace_execution(code_string)

# Example to show the new type-preserving print output
if __name__ == "__main__":
    test_code = """
x = 100
print(x)
print("---")
print("value is:", x)
"""
    
    print("\n=== DEMONSTRATING TYPE-PRESERVING PRINT OUTPUT ===")
    trace_output = trace_python_code(test_code)
    print(trace_output)