import asyncio
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
# Tool definitions
web_search_preview = WebSearchTool(
  search_context_size="medium",
  user_location={
    "type": "approximate"
  }
)
class ClassifierSchema(BaseModel):
  classification: str


class FlightAgentSchema(BaseModel):
  background: str
  flightNumber: str
  departureCity: str
  departureTime: str
  arrivalCity: str
  arrivalTime: str


classifier = Agent(
  name="Classifier",
  instructions="You are a helpful travel assistant for classifying wheater a message  is about an itinerary or a flight",
  model="gpt-4.1",
  output_type=ClassifierSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


flight_agent = Agent(
  name="Flight Agent",
  instructions="You are a travel assistant. Always recommend a specific flight to go to. USe airport codes.  Not include timeyones or am/pm. Choose a background color based on the destination",
  model="gpt-4.1",
  tools=[
    web_search_preview
  ],
  output_type=FlightAgentSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


itinerary_agent = Agent(
  name="Itinerary Agent",
  instructions="You are a travel assistant so build a concise intinerary. Output SHould be in HTMl",
  model="gpt-4.1",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
    with trace("Flight Agent"):
        workflow = workflow_input.model_dump()
        conversation_history = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": workflow["input_as_text"]}
                ],
            }
        ]

        # ---- Classifier ----
        classifier_result_temp = await Runner.run(
            classifier,
            input=conversation_history,
            run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
        )

        conversation_history.extend(
            [item.to_input_item() for item in classifier_result_temp.new_items]
        )

        classifier_result = classifier_result_temp.final_output.model_dump()
        classification = classifier_result["classification"]

        # ---- Branch ----
        if classification == "flight_info":
            agent_used = "flight_agent"
            agent_result_temp = await Runner.run(
                flight_agent,
                input=conversation_history,
                run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
            )
            agent_output = agent_result_temp.final_output.model_dump()

        else:
            agent_used = "itinerary_agent"
            agent_result_temp = await Runner.run(
                itinerary_agent,
                input=conversation_history,
                run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"}),
            )
            agent_output = agent_result_temp.final_output_as(str)

        # ---- Return structured result ----
        return {
            "agent_used": agent_used,
            "classification": classification,
            "output": agent_output,
        }

    with trace("Flight Agent"):
      state = {

      }
      workflow = workflow_input.model_dump()
      conversation_history: list[TResponseInputItem] = [
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": workflow["input_as_text"]
            }
          ]
        }
      ]
      classifier_result_temp = await Runner.run(
        classifier,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_6926bbf93dcc8190a0d120a2119a860a0215eea91f07937f"
        })
      )

      conversation_history.extend([item.to_input_item() for item in classifier_result_temp.new_items])

      classifier_result = {
        "output_text": classifier_result_temp.final_output.json(),
        "output_parsed": classifier_result_temp.final_output.model_dump()
      }
      if classifier_result["output_parsed"]["classification"] == "flight_info":
        flight_agent_result_temp = await Runner.run(
          flight_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_6926bbf93dcc8190a0d120a2119a860a0215eea91f07937f"
          })
        )

        conversation_history.extend([item.to_input_item() for item in flight_agent_result_temp.new_items])

        flight_agent_result = {
          "output_text": flight_agent_result_temp.final_output.json(),
          "output_parsed": flight_agent_result_temp.final_output.model_dump()
        }
      elif classifier_result["output_parsed"]["classification"] == "itinerary":
        itinerary_agent_result_temp = await Runner.run(
          itinerary_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_6926bbf93dcc8190a0d120a2119a860a0215eea91f07937f"
          })
        )

        conversation_history.extend([item.to_input_item() for item in itinerary_agent_result_temp.new_items])

        itinerary_agent_result = {
          "output_text": itinerary_agent_result_temp.final_output_as(str)
        }
      else:
        itinerary_agent_result_temp = await Runner.run(
          itinerary_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_6926bbf93dcc8190a0d120a2119a860a0215eea91f07937f"
          })
        )

        conversation_history.extend([item.to_input_item() for item in itinerary_agent_result_temp.new_items])

        itinerary_agent_result = {
          "output_text": itinerary_agent_result_temp.final_output_as(str)
        }
      
if __name__ == "__main__":
    print("=== Flight \ INtenrary Agent ===")
    user_text = input("Write your intenrary or question: ")

    result = asyncio.run(run_workflow(WorkflowInput(input_as_text=user_text)))

    print("\n===== RESULTADO =====")
    print(result)      
