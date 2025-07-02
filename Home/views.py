# C:\projects\Intellitrace\agent\Home\views.py

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
import json

# FIX: Use a relative import to find the tracer module in the same directory.
from .tracer import trace_python_code

@csrf_exempt
def run_tracer_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            code = data.get("code", "")

            if not code.strip():
                return JsonResponse({"error": "No code provided"}, status=400)

            # Get trace result. trace_python_code returns a JSON string.
            trace_json_string = trace_python_code(code, output_format="json")

            # FIX: Parse the JSON string into a Python object (a list)
            # before passing it to JsonResponse.
            trace_data = json.loads(trace_json_string)

            return JsonResponse(trace_data, safe=False)

        except Exception as e:
            # Provide a more specific error message in the response
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

    return JsonResponse({"error": "Only POST method allowed"}, status=405)

def index(request):
    return render(request, "index.html")