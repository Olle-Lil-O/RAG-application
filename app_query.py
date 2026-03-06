import gradio as gr
from chat_manager import ChatManagerWithTools
import dotenv
dotenv.load_dotenv(".env")

chat_manager = ChatManagerWithTools()
source_choices = ["(All sources)"] + chat_manager.list_available_sources()
source_dropdown = gr.Dropdown(
    source_choices,
    label="Source filter",
    value=source_choices[0] if source_choices else "",
    interactive=True,
)
retrieve_k_slider = gr.Slider(1, 50, value=20, step=1, label="Retriever k")
rerank_top_k_slider = gr.Slider(1, 20, value=10, step=1, label="Top k after rerank")
debug_checkbox = gr.Checkbox(label="Debug Mode", value=False)


async def chat_stream(message, history, source, retrieve_k, rerank_top_k, debug):
    normalized_source = None
    if source and source != "(All sources)":
        normalized_source = source
    chat_manager.set_source(normalized_source)
    chat_manager.set_retrieval_params(int(retrieve_k), int(rerank_top_k))

    response = ""
    async for chunk in chat_manager.stream_response(message, debug=debug):
        response += chunk
        yield response


app = gr.ChatInterface(
        fn=chat_stream,
    type="tuples",
        title="Books Knowledge Base.",
        autoscroll=True,
        theme="soft",
        additional_inputs=[source_dropdown, retrieve_k_slider, rerank_top_k_slider, debug_checkbox],
    )

if __name__ == "__main__":
    app.queue().launch(pwa=True)