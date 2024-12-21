import boto3
import json

prompt_data="""
about KL Rahul
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt": prompt_data,  # Use the raw prompt data
    "max_gen_len": 100,
    "temperature": 0.5,
    "top_p": 0.9
}
body = json.dumps(payload)

# Specify the model ID
model_id = "meta.llama3-70b-instruct-v1:0"

# Invoke the model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse and print the response
response_body = json.loads(response.get("body").read())
response_text = response_body.get("generation", "No response generated.")
print(response_text)
