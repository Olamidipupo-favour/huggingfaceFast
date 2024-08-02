from gradio_client import Client

client = Client("HuggingFaceM4/idefics2_playground")
result = client.predict(
		message={"text":"","files":[]},
		request="idefics2-8b-chatty",
		param_3="Greedy",
		param_4=0.4,
		param_5=512,
		param_6=1.1,
		param_7=0.8,
		api_name="/chat"
)
print(result)