import base64
import json
import time
from io import BytesIO
from typing import Annotated, List, Union

# Import Azure client and specific errors from the openai library
from openai import AzureOpenAI, APITimeoutError, RateLimitError, BadRequestError
import PIL
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

# --- Placeholder Definitions ---
# Assuming BaseService and Block are defined elsewhere.
# If not, you can use these simple placeholders or import your actual definitions.
# from marker.schema.blocks import Block  # Uncomment if you have this structure
# from marker.services import BaseService # Uncomment if you have this structure


class AzureOpenAIService(BaseService):
    """
    A service class to interact with Azure OpenAI's chat completion API,
    specifically designed for models that support vision (image input) and JSON output.

    Inherits configuration like max_retries and timeout from BaseService.
    """
    # --- Azure Specific Configuration ---
    # These fields must be configured when creating an instance of the service.
    azure_endpoint: Annotated[
        str,
        "The Azure OpenAI endpoint URL (e.g., https://your-resource-name.openai.azure.com/)"
    ] = None
    azure_api_key: Annotated[
        str,
        "The API key for your Azure OpenAI service."
    ] = None
    azure_deployment_name: Annotated[
        str,
        "The deployment name of the vision-capable model in Azure OpenAI (e.g., 'gpt-4o-deployment')."
    ] = None
    azure_api_version: Annotated[
        str,
        "The API version for the Azure OpenAI service (e.g., '2024-05-01-preview'). Check Azure docs for compatible versions."
    ] = "2024-05-01-preview" # Default to a recent preview version, adjust if needed

    # --- Helper Methods ---
    def image_to_base64(self, image: PIL.Image.Image, format="WEBP", quality=95) -> str:
        """
        Converts a PIL Image object to a base64 encoded string.

        Args:
            image: The PIL.Image.Image object.
            format: The image format to save as before encoding (e.g., "WEBP", "PNG", "JPEG"). WEBP is often efficient.
            quality: The quality setting for formats like WEBP/JPEG (1-100).

        Returns:
            A base64 encoded string representing the image.
        """
        image_bytes = BytesIO()
        try:
            image.save(image_bytes, format=format, quality=quality)
            return base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return "" # Return empty string on error

    def prepare_images(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> List[dict]:
        """
        Prepares a list of PIL Images for the Azure OpenAI API request format.

        Args:
            images: A single PIL Image or a list of PIL Images.

        Returns:
            A list of dictionaries, each representing an image in the format expected by the API.
        """
        if isinstance(images, Image.Image):
            images = [images] # Ensure it's a list

        prepared_image_data = []
        for img in images:
            base64_string = self.image_to_base64(img, format="WEBP") # Using WEBP
            if base64_string: # Only add if conversion was successful
                prepared_image_data.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            # Format: data:[mime_type];base64,[base64_string]
                            "url": f"data:image/webp;base64,{base64_string}",
                            # Optional detail parameter: "low", "high", "auto" (default)
                            # "detail": "auto"
                        }
                    }
                )
            else:
                print("Warning: Skipping an image due to conversion error.")
        return prepared_image_data

    # --- Core API Call Method ---
    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image],
        block: Block, # Assumes Block has an update_metadata method
        response_schema: type[BaseModel], # Pydantic model for expected JSON response
        max_retries: int | None = 8,
        timeout: int | None = None,
        max_tokens: int | None = None, # Max tokens for the completion
    ) -> dict:
        """
        Makes a call to the configured Azure OpenAI deployment with the given
        prompt and image(s). It expects a JSON response conforming to the
        provided Pydantic response_schema.

        Args:
            prompt: The text prompt to send to the model.
            image: A single PIL Image or a list of PIL Images.
            block: An object (like the placeholder Block) with an `update_metadata` method
                   to record token usage and request counts.
            response_schema: The Pydantic model class defining the expected structure
                             of the JSON response.
            max_retries: Override the default max retries for this call.
            timeout: Override the default timeout for this call.
            max_tokens: Max tokens to generate in the response.

        Returns:
            A dictionary representing the parsed and validated JSON response,
            or an empty dictionary if the call fails after retries or encounters
            an unrecoverable error.
        """
        
        # Set effective retries and timeout
        current_max_retries = max_retries if max_retries is not None else self.max_retries
        current_timeout = timeout if timeout is not None else self.timeout

        # Ensure image is a list
        if not isinstance(image, list):
            image = [image]

        # Initialize the Azure OpenAI client
        client = self.get_client()
        if not client:
            print("Error: Azure OpenAI client initialization failed. Cannot proceed.")
            # Record request failure if possible (optional)
            # block.update_metadata(llm_request_count=1, llm_error="Client Initialization Failed")
            return {} # Return empty dict indicating failure

        # Prepare image data for the payload
        image_data = self.prepare_images(image)
        if not image_data and image: # Check if image was provided but preparation failed
             print("Warning: Image(s) provided but failed to prepare them for the API.")
             # Decide if you want to proceed without images or fail
             # return {} # Option: fail if images are crucial and failed preparation

        # Construct the messages payload for the API call
        # Following the pattern: text prompt first, then all images.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_data, # Spread the list of image dictionaries
                ],
            }
            # Optional: Add a system prompt if needed for context or instructions
            # {"role": "system", "content": "You are an AI assistant..."}
        ]

        json_schema_supported = True
        if "Checklist-GPT-4o-0513" in self.azure_deployment_name:
            json_schema_supported = False
        # --- API Call with Retry Logic ---
        tries = 0
        while tries < current_max_retries:
            try:
                if json_schema_supported:
                    try:
                        # print(f"Attempt {tries + 1}/{current_max_retries}: Calling Azure OpenAI deployment '{self.azure_deployment_name}'...")
                        # Make the API call using the standard 'parse' method
                        # if "o3" in self.azure_deployment_name or "o1" in self.azure_deployment_name:
                        response = client.beta.chat.completions.parse(
                            model=self.azure_deployment_name, # Specify the deployment name
                            messages=messages,
                            max_completion_tokens=max_tokens,
                            # temperature=temperature,
                            # Request JSON output explicitly. The model must support this.
                            response_format=response_schema,#{"type": "json_object"},
                            timeout=current_timeout,
                            # Add custom headers if needed (e.g., for tracking)
                            extra_headers={
                                "X-Title": "Marker-Azure", # Example header
                                "HTTP-Referer": "https://github.com/VikParuchuri/marker", # Example header
                            },
                        )

                        # else:
                        #     response = client.beta.chat.completions.parse(
                        #         model=self.azure_deployment_name, # Specify the deployment name
                        #         messages=messages,
                        #         max_tokens=max_tokens,
                        #         temperature=temperature,
                        #         # Request JSON output explicitly. The model must support this.
                        #         response_format=response_schema, #{"type": "json_object"},
                        #         timeout=current_timeout,
                        #         # Add custom headers if needed (e.g., for tracking)
                        #         extra_headers={
                        #             "X-Title": "Marker-Azure", # Example header
                        #             "HTTP-Referer": "https://github.com/VikParuchuri/marker", # Example header
                        #         },
                        #     )
                    except Exception as e:
                        print(e)
                        if "'response_format' of type 'json_schema' is not supported with this model." not in str(e):
                            break
                        print("Json_schema not supported. Using JSON mode instead...")
                        json_schema_supported = False
                        json_messages = messages.copy()
                        for json_message in json_messages:
                            for content in json_message["content"]:
                                if content["type"] == "text":
                                    content["text"] = content["text"] + "\nReturn the output in JSON format:\n" + response_schema.get_description()
                        
                        response = client.beta.chat.completions.parse(
                                model=self.azure_deployment_name, # Specify the deployment name
                                messages=json_messages,
                                max_completion_tokens=max_tokens,
                                # temperature=temperature,
                                # Request JSON output explicitly. The model must support this.
                                response_format={ "type": "json_object" }, #{"type": "json_object"},
                                timeout=current_timeout,
                                # Add custom headers if needed (e.g., for tracking)
                                extra_headers={
                                    "X-Title": "Marker-Azure", # Example header
                                    "HTTP-Referer": "https://github.com/VikParuchuri/marker", # Example header
                                },
                            )

                else:
                    json_messages = messages.copy()
                    for json_message in json_messages:
                            for content in json_message["content"]:
                                if content["type"] == "text":
                                    content["text"] = content["text"] + "\nReturn the output in JSON format:\n" + response_schema.get_description()
                    response = client.beta.chat.completions.parse(
                        model=self.azure_deployment_name, # Specify the deployment name
                        messages=json_messages,
                        max_completion_tokens=max_tokens,
                        # temperature=temperature,
                        # Request JSON output explicitly. The model must support this.
                        response_format={ "type": "json_object" }, #{"type": "json_object"},
                        timeout=current_timeout,
                        # Add custom headers if needed (e.g., for tracking)
                        extra_headers={
                            "X-Title": "Marker-Azure", # Example header
                            "HTTP-Referer": "https://github.com/VikParuchuri/marker", # Example header
                        },
                    ) 
                # --- Process Successful Response ---
                # Extract the response content (should be a JSON string)
                response_content = response.choices[0].message.content
                # Get token usage if available
                total_prompt_tokens = response.usage.prompt_tokens
                total_completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Update metadata (token count, request count)
                block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                print(f"Azure API call successful. Total tokens used: {total_tokens} (prompt: {total_prompt_tokens}, completion: {total_completion_tokens}).")

                # Parse the JSON string and validate with the Pydantic schema
                try:
                    parsed_response = response_schema.model_validate_json(response_content)
                    # Return the validated data as a dictionary
                    return parsed_response.model_dump()
                except json.JSONDecodeError as json_err:
                    print(f"Error: Failed to decode JSON response from API: {json_err}")
                    print(f"Received content: {response_content}")
                    # Treat as failure for this attempt, maybe retry if appropriate
                    # Or break if JSON is consistently malformed
                    # block.update_metadata(llm_error="JSONDecodeError")
                    tries += 1 # Count as a failed attempt
                    if tries >= current_max_retries: break # Don't retry if max retries reached
                    # No automatic retry here, could add a short sleep if desired
                    continue # Go to next retry loop iteration
                except Exception as validation_err: # Catch Pydantic validation errors etc.
                    print(f"Error: Response validation failed against schema '{response_schema.__name__}': {validation_err}")
                    print(f"Received content: {response_content}")
                    # Treat as failure, break the loop as the response structure is wrong
                    # block.update_metadata(llm_error="ValidationError")
                    break # Exit loop, validation failed


            # --- Handle Retriable Errors ---
            except (APITimeoutError, RateLimitError) as e:
                tries += 1
                # block.update_metadata(llm_request_count=1, llm_error=type(e).__name__) # Record error type
                if tries >= current_max_retries:
                    print(f"Error: Max retries ({current_max_retries}) reached after {type(e).__name__}. Aborting.")
                    break # Exit loop
                # Exponential backoff: wait 2^tries seconds (2, 4, 8, ...)
                wait_time = (2 ** tries)
                print(
                    f"Error: {type(e).__name__}: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{current_max_retries})"
                )
                time.sleep(wait_time)
                # Continue to the next iteration of the while loop

            # --- Handle Other Non-Retriable API Errors ---
            except Exception as e:
                # Includes AuthenticationError, PermissionDeniedError, InvalidRequestError, etc.
                print(f"An unexpected or non-retriable error occurred: {type(e).__name__}: {e}")
                # Log the full traceback for detailed debugging if needed
                # import traceback
                # print(traceback.format_exc())
                # block.update_metadata(llm_request_count=1, llm_error=type(e).__name__)
                break # Exit loop on unexpected/fatal error

        # --- Return Empty Dict if all retries fail or an unrecoverable error occurred ---
        print("Failed to get a valid response after all retries or due to an error.")
        # block.update_metadata(llm_success=False) # Mark final status as failure
        return {}

    # --- Client Initialization Method ---
    def get_client(self) -> AzureOpenAI | None:
        """
        Initializes and returns the AzureOpenAI client using the configured
        endpoint, API key, and API version.

        Returns:
            An instance of openai.AzureOpenAI, or None if configuration is
            missing or initialization fails.
        """
        # Validate that essential configuration parameters are set
        if not all([self.azure_endpoint, self.azure_api_key, self.azure_api_version, self.azure_deployment_name]):
             print("Error: Azure OpenAI configuration is incomplete. Missing one or more of: "
                   "azure_endpoint, azure_api_key, azure_api_version, azure_deployment_name.")
             return None # Return None if configuration is missing

        try:
            # Initialize and return the AzureOpenAI client
            # print(f"Initializing AzureOpenAI client for endpoint: {self.azure_endpoint}, API Version: {self.azure_api_version}")
            client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                # Note: The deployment name is passed during the 'create' call, not here.
            )
            return client
        except Exception as e:
            # Catch potential errors during client initialization
            print(f"Error initializing Azure OpenAI client: {e}")
            return None # Return None on initialization failure


# --- Example Usage (Optional: Uncomment and modify to test) ---
# if __name__ == "__main__":
#     # --- Configuration ---
#     # IMPORTANT: Replace with your actual Azure credentials and deployment details
#     # Consider using environment variables or a config file for security.
#     AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "YOUR_ENDPOINT_HERE")
#     AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "YOUR_API_KEY_HERE")
#     AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "YOUR_DEPLOYMENT_NAME_HERE") # e.g., your gpt-4o deployment
#     AZURE_API_VERSION = "2024-05-01-preview" # Check Azure docs for the latest/best version
#
#     # --- Define a Pydantic model for the expected JSON response structure ---
#     class ImageDescriptionResponse(BaseModel):
#         description: Annotated[str, "A textual description of the image content."]
#         identified_objects: Annotated[List[str], "A list of objects identified in the image."]
#         confidence_score: Annotated[float, "A score indicating confidence in the description."] | None = None
#
#     # --- Initialize the Azure OpenAI Service ---
#     azure_service = AzureOpenAIService(
#         azure_endpoint=AZURE_ENDPOINT,
#         azure_api_key=AZURE_API_KEY,
#         azure_deployment_name=AZURE_DEPLOYMENT_NAME,
#         azure_api_version=AZURE_API_VERSION,
#         timeout=180 # Optional: Increase timeout for potentially long calls
#     )
#
#     # --- Prepare Inputs ---
#     # The prompt should guide the model to respond in the desired JSON format
#     prompt_text = """
#     Analyze the provided image and respond ONLY with a JSON object matching the following structure:
#     {
#       "description": "A concise description of the main subject and scene.",
#       "identified_objects": ["object1", "object2", ...],
#       "confidence_score": 0.0-1.0 (optional)
#     }
#     Describe the image content accurately.
#     """
#     image_path = "path/to/your/test_image.png" # <--- CHANGE THIS to a valid image path
#
#     try:
#         # Load the image using PIL
#         img = Image.open(image_path)
#     except FileNotFoundError:
#         print(f"Error: Test image file not found at '{image_path}'. Please provide a valid path.")
#         img = None
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         img = None
#
#     # Create a placeholder block object to track metadata
#     dummy_block = Block()
#
#     # --- Make the API Call ---
#     if img and azure_service.get_client(): # Proceed only if image loaded and client is ready
#         print(f"\n--- Calling Azure OpenAI Service for image: {image_path} ---")
#         result_dict = azure_service(
#             prompt=prompt_text,
#             image=img, # Can also pass a list: [img1, img2]
#             block=dummy_block,
#             response_schema=ImageDescriptionResponse # Pass the Pydantic model class
#         )
#
#         # --- Print Result ---
#         print("\n--- API Response ---")
#         if result_dict:
#             print(json.dumps(result_dict, indent=2))
#         else:
#             print("API call failed or returned no valid data.")
#
#         print("\n--- Final Block Metadata ---")
#         print(json.dumps(dummy_block.metadata, indent=2))
#
#     else:
#         if not img:
#             print("Skipping API call because the image could not be loaded.")
#         if not azure_service.get_client():
#              print("Skipping API call because the Azure client could not be initialized (check config).")

