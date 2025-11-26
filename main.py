import asyncio
import webbrowser
from pathlib import Path
from pydantic import BaseModel
from agents import (
    WebSearchTool,
    Agent,
    ModelSettings,
    TResponseInputItem,
    Runner,
    RunConfig,
    trace,
)


# -----------------------------
#        SCHEMAS
# -----------------------------
class ClassifierSchema(BaseModel):
    classification: str


class FlightAgentSchema(BaseModel):
    background: str
    flightNumber: str
    departureCity: str
    departureTime: str
    arrivalCity: str
    arrivalTime: str


class WorkflowInput(BaseModel):
    input_as_text: str



# -----------------------------
#        AGENTS
# -----------------------------
web_search_preview = WebSearchTool(
    search_context_size="medium",
    user_location={"type": "approximate"},
)

classifier = Agent(
    name="Classifier",
    instructions="You classify whether the message is about a flight_info or an itinerary.",
    model="gpt-4.1",
    output_type=ClassifierSchema,
    model_settings=ModelSettings(
        temperature=1, top_p=1, max_tokens=2048, store=True
    ),
)

flight_agent = Agent(
    name="Flight Agent",
    instructions="Return structured flight info.",
    model="gpt-4.1",
    tools=[web_search_preview],
    output_type=FlightAgentSchema,
    model_settings=ModelSettings(
        temperature=1, top_p=1, max_tokens=2048, store=True
    ),
)

itinerary_agent = Agent(
    name="Itinerary Agent",
    instructions="Return a concise travel itinerary.",
    model="gpt-4.1",
    model_settings=ModelSettings(
        temperature=1, top_p=1, max_tokens=2048, store=True
    ),
)



# -----------------------------
#        LOGGING UTILS
# -----------------------------
def log(msg: str):
    print(f"[LOG] {msg}")

def log_section(msg: str):
    print("\n===== " + msg + " =====")



# -----------------------------
#   HTML RENDER HELPERS
# -----------------------------
def render_flight_widget(data: dict) -> str:
    """Builds an HTML widget that looks close to the OpenAI Widget Studio design."""
    return f"""
<html>
<head>
<style>
    body {{
        font-family: Arial;
        background: white;
        padding: 30px;
    }}
    .card {{
        width: 350px;
        padding: 20px;
        border-radius: 20px;
        color: white;
        background: #996699;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }}
    .title {{
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align:center;
    }}
    .label {{
        opacity: 0.8;
        font-size: 12px;
    }}
    .value {{
        font-size: 20px;
        font-weight: 600;
    }}
</style>
</head>
<body>
<div class="card">
    <div class="title">{data["flightNumber"]}</div>

    <div>
        <div class="label">From</div>
        <div class="value">{data["departureCity"]} - {data["departureTime"]}</div>
    </div>

    <br/>

    <div>
        <div class="label">To</div>
        <div class="value">{data["arrivalCity"]} - {data["arrivalTime"]}</div>
    </div>
</div>
</body>
</html>
"""


def render_itinerary_widget(text: str) -> str:
    return f"""
<html>
<head>
<style>
    body {{
        font-family: Arial;
        background: #fafafa;
        padding: 30px;
    }}
    .box {{
        width: 500px;
        padding: 20px;
        border-radius: 10px;
        background: #89cff0;
        border: 1px solid #ddd;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    pre {{
        white-space: pre-wrap;
        font-size: 16px;
    }}
</style>
</head>
<body>
<div class="box">
    <h2>Suggested Itinerary</h2>
    <pre>{text}</pre>
</div>
</body>
</html>
"""


def open_widget_in_browser(html_content: str, filename: str = "widget_output.html"):
    path = Path(filename)
    path.write_text(html_content, encoding="utf-8")
    webbrowser.open("file://" + str(path.resolve()))
    log(f"HTML widget written to {path.resolve()}")



# -----------------------------
#   WORKFLOW EXECUTION
# -----------------------------
async def classify(conversation_history):
    log("Running classifier agent...")
    return await Runner.run(
        classifier,
        input=conversation_history,
        run_config=RunConfig(trace_metadata={"source": "workflow"}),
    )


async def run_flight_agent(history):
    log("Running flight agent...")
    return await Runner.run(
        flight_agent,
        input=history,
        run_config=RunConfig(trace_metadata={"source": "workflow"}),
    )


async def run_itinerary_agent(history):
    log("Running itinerary agent...")
    return await Runner.run(
        itinerary_agent,
        input=history,
        run_config=RunConfig(trace_metadata={"source": "workflow"}),
    )



async def run_workflow(workflow_input: WorkflowInput):
    log_section("Starting Workflow")

    conversation_history = [
        {"role": "user", "content": [{"type": "input_text", "text": workflow_input.input_as_text}]}
    ]
    

    # ---- CLASSIFICATION ----
    classifier_result_temp = await classify(conversation_history)
    classification = classifier_result_temp.final_output.model_dump()["classification"]
    log(f"Classifier result: {classification}")

    conversation_history.extend(i.to_input_item() for i in classifier_result_temp.new_items)

    # ---- BRANCH ----
    if classification == "flight_info":
        flight_temp = await run_flight_agent(conversation_history)
        output_data = flight_temp.final_output.model_dump()

        log("Rendering flight widget...")
        html = render_flight_widget(output_data)
        open_widget_in_browser(html)

        return {
            "agent": "flight_agent",
            "classification": classification,
            "output": output_data,
        }

    else:
        itin_temp = await run_itinerary_agent(conversation_history)
        output_text = itin_temp.final_output_as(str)

        log("Rendering itinerary widget...")
        html = render_itinerary_widget(output_text)
        open_widget_in_browser(html)

        return {
            "agent": "itinerary_agent",
            "classification": classification,
            "output": output_text,
        }



# -----------------------------
#        ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    print("=== Flight / Itinerary Agent ===")
    user_text = input("Write your itinerary or question: ")

    result = asyncio.run(
        run_workflow(WorkflowInput(input_as_text=user_text))
    )

    log_section("FINAL RESULT")
    print(result)
