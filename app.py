import os
import sys
import io
import traceback

from typing import TypedDict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

from langserve import add_routes


# ============================================================
# 1. ENVIRONMENT / GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# You can change this in Render Environment Variables.
#
# Example:
# GEMINI_MODEL=gemini-3.5-flash
#
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash"
)


# ============================================================
# 2. GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

print(
    f"Gemini initialized with model: {GEMINI_MODEL}"
)


# ============================================================
# 3. STATE
# ============================================================

class CrewState(TypedDict, total=False):

    # Original user task
    task: str

    # Developer output
    code: str

    # Tester output
    test_cases: str

    # Execution output
    execution_output: str

    # Final tester report
    report: str

    # Manager decision
    decision: str


# ============================================================
# 4. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Developer Tester Manager",
    description=(
        "LangGraph workflow using Gemini with "
        "Developer -> Tester -> Manager state flow."
    ),
    version="1.0.0"
)


# ============================================================
# 5. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "AI Developer Tester Manager",
        "model": GEMINI_MODEL
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 7. HELPER
# ============================================================

def extract_text(content) -> str:
    """
    Convert Gemini response content into a normal string.
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:
                    parts.append(
                        str(item["text"])
                    )

            else:

                parts.append(
                    str(item)
                )

        return "".join(parts)

    return str(content)


# ============================================================
# 8. PYTHON EXECUTION TOOL
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code and return stdout or traceback.
    """

    if not isinstance(code, str):

        code = str(code)


    # Remove Markdown fences.
    clean_code = (
        code
        .replace("```python", "")
        .replace("```Python", "")
        .replace("```", "")
        .strip()
    )


    old_stdout = sys.stdout

    new_stdout = io.StringIO()

    sys.stdout = new_stdout


    try:

        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()


    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )


    finally:

        sys.stdout = old_stdout


    result = result.strip()


    if result:

        return result


    return "Success (no terminal output)"


# ============================================================
# 9. TEST CASE GENERATOR
# ============================================================

@tool
def generate_test_cases(
    task_description: str
) -> str:
    """
    Generate QA test scenarios.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for this Python coding task:

{task_description}

Requirements:

1. Include normal cases.
2. Include edge cases.
3. Include invalid input cases when appropriate.
4. Explain what each test verifies.
5. Return ONLY a numbered list.
"""


    response = llm.invoke(prompt)

    return extract_text(
        response.content
    )


# ============================================================
# 10. DEVELOPER NODE
# ============================================================

def developer_node(
    state: CrewState
):

    print("\n")
    print("=" * 70)
    print("DEVELOPER NODE")
    print("=" * 70)


    task = state.get("task")


    if not task:

        raise ValueError(
            "Developer did not receive a task."
        )


    print("\nTask received:")
    print(task)


    # --------------------------------------------------------
    # Developer prompt
    # --------------------------------------------------------

    prompt = f"""
You are a Senior Python Developer.

Solve this coding task:

{task}

Requirements:

- Write clean Python code.
- Make it executable.
- Handle reasonable edge cases.
- Return ONLY Python source code.
- Do NOT return Markdown.
- Do NOT return explanations.
- Do NOT use ```python.
"""


    print(
        "\nCalling Gemini Developer..."
    )


    response = llm.invoke(prompt)


    code = extract_text(
        response.content
    )


    # --------------------------------------------------------
    # Clean Gemini output
    # --------------------------------------------------------

    code = (
        code
        .replace("```python", "")
        .replace("```Python", "")
        .replace("```", "")
        .strip()
    )


    if not code:

        raise ValueError(
            "Developer generated empty code."
        )


    print("\nDeveloper generated code:")
    print("-" * 70)
    print(code)
    print("-" * 70)


    # --------------------------------------------------------
    # STATE UPDATE
    # --------------------------------------------------------

    updated_state = {
        "code": code
    }


    print(
        "\nDeveloper -> Tester"
    )

    print(
        "State contains code:",
        bool(updated_state["code"])
    )


    return updated_state


# ============================================================
# 11. TESTER NODE
# ============================================================

def tester_node(
    state: CrewState
):

    print("\n")
    print("=" * 70)
    print("TESTER NODE")
    print("=" * 70)


    # --------------------------------------------------------
    # Receive state from Developer
    # --------------------------------------------------------

    task = state.get("task")

    code = state.get("code")


    if not task:

        raise ValueError(
            "Tester did not receive task."
        )


    if not code:

        raise ValueError(
            "Tester did not receive code "
            "from Developer."
        )


    print(
        "\nTester received Developer code."
    )


    print("-" * 70)
    print(code)
    print("-" * 70)


    # ========================================================
    # Generate tests
    # ========================================================

    print(
        "\nGenerating test cases..."
    )


    test_cases = generate_test_cases.invoke(
        task
    )


    test_cases = extract_text(
        test_cases
    )


    # ========================================================
    # Execute code
    # ========================================================

    print(
        "\nExecuting Developer code..."
    )


    execution_output = run_python_code.invoke(
        {
            "code": code
        }
    )


    # ========================================================
    # Build report
    # ========================================================

    report = f"""
### EXECUTION OUTPUT

{execution_output}


### TEST SCENARIOS

{test_cases}
"""


    print(
        "\nTester generated report:"
    )

    print("-" * 70)

    print(report)

    print("-" * 70)


    # --------------------------------------------------------
    # STATE UPDATE
    # --------------------------------------------------------

    updated_state = {

        "test_cases": test_cases,

        "execution_output": execution_output,

        "report": report
    }


    print(
        "\nTester -> Manager"
    )

    print(
        "State contains report:",
        bool(updated_state["report"])
    )


    return updated_state


# ============================================================
# 12. MANAGER NODE
# ============================================================

def manager_node(
    state: CrewState
):

    print("\n")
    print("=" * 70)
    print("MANAGER NODE")
    print("=" * 70)


    # --------------------------------------------------------
    # Receive Tester state
    # --------------------------------------------------------

    report = state.get("report")

    code = state.get("code")

    execution_output = state.get(
        "execution_output"
    )


    if not report:

        raise ValueError(
            "Manager did not receive Tester report."
        )


    print(
        "\nManager received Tester report."
    )


    print("-" * 70)

    print(report)

    print("-" * 70)


    # ========================================================
    # Manager evaluates execution
    # ========================================================

    manager_prompt = f"""
You are a Senior Engineering Manager.

Review the following Python implementation
and its execution result.

CODE:

{code}


EXECUTION OUTPUT:

{execution_output}


TEST REPORT:

{report}


Decide whether the implementation appears
to have passed testing.

Return ONLY one of:

APPROVED

or

NEEDS_REVISION
"""


    response = llm.invoke(
        manager_prompt
    )


    decision = extract_text(
        response.content
    ).strip()


    # --------------------------------------------------------
    # Normalize decision
    # --------------------------------------------------------

    if "APPROVED" in decision.upper():

        decision = "APPROVED"

    else:

        decision = "NEEDS_REVISION"


    print(
        "\nManager decision:",
        decision
    )


    # --------------------------------------------------------
    # STATE UPDATE
    # --------------------------------------------------------

    return {
        "decision": decision
    }


# ============================================================
# 13. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(
    CrewState
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "manager",
    manager_node
)


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    "manager"
)

workflow.add_edge(
    "manager",
    END
)


# ============================================================
# 14. COMPILE
# ============================================================

graph = workflow.compile()


print("\n")
print("=" * 70)
print("LANGGRAPH COMPILED")
print("=" * 70)

print(
    """
WORKFLOW:

API
 |
 v
Developer
 |
 | code
 v
Tester
 |
 | test_cases
 | execution_output
 | report
 v
Manager
 |
 | decision
 v
END
"""
)


# ============================================================
# 15. DIRECT FASTAPI ENDPOINT
# ============================================================

@app.post("/run")
async def run_workflow(
    request: dict
):

    # --------------------------------------------------------
    # Get task from JSON
    # --------------------------------------------------------

    task = request.get("task")


    if not task:

        return {
            "success": False,
            "error": (
                "Missing 'task' in request body."
            )
        }


    # --------------------------------------------------------
    # Initial LangGraph state
    # --------------------------------------------------------

    initial_state: CrewState = {

        "task": task,

        "code": "",

        "test_cases": "",

        "execution_output": "",

        "report": "",

        "decision": ""
    }


    try:

        # ----------------------------------------------------
        # Execute LangGraph
        # ----------------------------------------------------

        final_state = graph.invoke(
            initial_state
        )


        # ----------------------------------------------------
        # Return useful API response
        # ----------------------------------------------------

        return {

            "success": True,

            "task": final_state.get(
                "task"
            ),

            "code": final_state.get(
                "code"
            ),

            "test_cases": final_state.get(
                "test_cases"
            ),

            "execution_output": final_state.get(
                "execution_output"
            ),

            "report": final_state.get(
                "report"
            ),

            "decision": final_state.get(
                "decision"
            )
        }


    except Exception as e:

        print(
            "\nWorkflow execution error:"
        )

        traceback.print_exc()


        return {

            "success": False,

            "error": str(e)
        }


# ============================================================
# 16. LANGSERVE ROUTE
# ============================================================

# This exposes the LangGraph as a LangServe API.
#
# Example:
#
# POST /workflow/invoke
#
# {
#     "input": {
#         "task": "Create a Python program..."
#     }
# }
#

add_routes(
    app,
    graph,
    path="/workflow"
)


# ============================================================
# 17. LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
