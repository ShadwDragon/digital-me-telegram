#!/usr/bin/env python3
"""
Simple interactive testing script for ULZII model
"""

import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from loguru import logger

class InteractiveModelTester:
    def __init__(
        self,
        lora_path: str = "./weights/LoRA/",
        base_model: str = "alpindale/Mistral-7B-v0.2-hf",
        target_user: str = "ULZII"
    ):
        """Initialize the model tester."""
        
        self.target_user = target_user
        logger.info(f"Loading model for {target_user}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            ),
            device_map={"": 0} if torch.cuda.is_available() else "auto",
        )
        
        # Load LoRA weights
        try:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info("✅ LoRA weights loaded successfully")
        except Exception as e:
            logger.warning(f"❌ Could not load LoRA weights: {e}")
            logger.info("Using base model only")
        
        self.model.eval()
        logger.info("Model ready for testing!")
    
    def generate_response(
        self, 
        context: str, 
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate a response from the model."""
        
        inputs = self.tokenizer(
            context, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def chat_interactive(self, username: str = None):
        """Start interactive chat session."""
        
        if username is None:
            username = "User"
        
        logger.info(f"Starting interactive chat between {username} and {self.target_user}")
        print(f"\n🤖 Chat with {self.target_user}!")
        print(f"You are chatting as: {username}")
        print("Type 'quit' to exit, 'clear' to clear conversation history\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input(f"{username}: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("Conversation history cleared.\n")
                    continue
                
                if not user_input:
                    continue
                
                # Build context with conversation history
                context_parts = []
                
                # Add recent conversation history (last 6 messages)
                for msg in conversation_history[-6:]:
                    context_parts.append(f"<|im_start|>{msg['user']}\n{msg['text']}<|im_end|>")
                
                # Add current message
                context_parts.append(f"<|im_start|>{username}\n{user_input}<|im_end|>")
                context_parts.append(f"<|im_start|>{self.target_user}\n")
                
                context = "\n".join(context_parts)
                
                # Generate response
                response = self.generate_response(context)
                
                print(f"{self.target_user}: {response}\n")
                
                # Update conversation history
                conversation_history.append({"user": username, "text": user_input})
                conversation_history.append({"user": self.target_user, "text": response})
                
                # Keep history manageable (last 20 messages)
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print("Sorry, there was an error. Please try again.")
    
    def single_response(self, username: str, message: str) -> str:
        """Generate a single response to a message."""
        
        context = f"<|im_start|>{username}\n{message}<|im_end|>\n<|im_start|>{self.target_user}\n"
        response = self.generate_response(context)
        return response


def main(
    lora_path: str = "./weights/LoRA/",
    base_model: str = "alpindale/Mistral-7B-v0.2-hf",
    target_user: str = "ULZII",
    username: str = None,
    message: str = None,
    interactive: bool = True
):
    """
    Interactive testing for your fine-tuned model.
    
    Args:
        lora_path: Path to LoRA weights
        base_model: Base model name
        target_user: Name of the user the model emulates
        username: Username for the conversation
        message: Single message to send (if provided, skips interactive mode)
        interactive: Whether to start interactive chat (ignored if message is provided)
    
    Examples:
        # Interactive chat
        python test_interactive.py --username "Erbold"
        
        # Single message
        python test_interactive.py --username "Erbold" --message "bro hezee tokyo irne gelee?"
        
        # Use different model paths
        python test_interactive.py --lora_path "./my_model/" --username "John"
    """
    
    # Initialize tester
    tester = InteractiveModelTester(lora_path, base_model, target_user)
    
    if message:
        # Single message mode
        if not username:
            username = "User"
        
        print(f"{username}: {message}")
        response = tester.single_response(username, message)
        print(f"{target_user}: {response}")
        
    elif interactive:
        # Interactive chat mode
        tester.chat_interactive(username)
    
    else:
        logger.info("No message provided and interactive mode disabled. Use --message or --interactive=True")


if __name__ == "__main__":
    fire.Fire(main)
