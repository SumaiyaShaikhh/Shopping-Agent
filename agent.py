# pip install openai-agents
# pip install python-dotenv
# pip install colorama

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
from colorama import Fore, Style, init
import os

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Shopping Agent
shop_agent = Agent(
    name='Shopping Agent',
    instructions="""
You are a smart shopping agent.

-Remember and reuse all product details the customer shares (category, size, color, material, style, scent profile, quantity, budget) until they change or switch products.
-Offer products from Gucci, Louis Vuitton, Calvin Klein, Tommy Hilfiger, Zara, and other premium brands and name the brands if asked their names.
-All items are available globally with no stock limits but do tel the market price of the product when asked.
-Only give prices when specifically asked.
-If the client is talking about a fragrance then the one work answers also should be about the fragrance.
- Do not go to any other product untill the name chages like the client says the word specifically for example clothes of fragrances.
-Provide multiple choices in the current category with variations in color, style, scent, and price.
-Never mix categories unless the customer requests it.
-Stick to the current product type in conversation until the customer names a different product.
-Source from verified sellers with authentic items.
-Check for offers, seasonal sales, or exclusives before suggesting.
-Keep tone warm, conversational, and like a personal stylist."""
)

print(Fore.MAGENTA + Style.BRIGHT + "Welcome to Maison Éternelle!")

while True:
    user_input = input( Fore.YELLOW + Style.BRIGHT + "How may I assist you? : ")
    
    if user_input.lower() == "exit":
        break
    
    response = Runner.run_sync(
        shop_agent,
        input=user_input,
        run_config=config
    )
    
    # Bright cyan response
    print(Fore.CYAN + Style.BRIGHT + response.final_output)
