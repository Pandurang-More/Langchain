from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id='HuggingFaceTB/SmolLM2-360M-Instruct',
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
        return_full_text=False
    )
    
    ) 
model =ChatHuggingFace(llm=llm) 

result=model.invoke("what is the capital of india")
print(result.content)