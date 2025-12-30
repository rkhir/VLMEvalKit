"""
Coconut Vision wrapper for Qwen2.5 VL models.

This implements the Meta Coconut continuous thought approach for Qwen2.5 VL,
allowing the model to perform latent reasoning before generating responses.
"""

import torch
import logging
import string

from vlmeval.vlm.base import BaseModel
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE

from vlmeval.vlm.coconut_vlm.qwen_vlmcoconut import QwenVLMCoconut


class CoconutQwenVision(BaseModel):
    """
    Coconut-enhanced Qwen2.5 VL model for continuous latent reasoning.
    
    This wraps Qwen2.5 VL with the Coconut mechanism that allows the model
    to perform multiple "thought" passes using latent tokens before generating
    the final response.
    """
    
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path='Qwen/Qwen2.5-VL-7B-Instruct',
        c_thought=2,
        min_pixels=None,
        max_pixels=None,
        **kwargs
    ):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using CoconutQwenVision.')
            raise e

        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except Exception as e:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise e

        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Load base Qwen2.5 VL model
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation='flash_attention_2',
        ).eval()
        self.device = 'cuda'
        
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Setup tokenizer and special tokens
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Add Coconut special tokens
        special_tokens = ["<|latent|>"]
        self.processor.tokenizer.add_tokens(special_tokens)

        self.latent_id = self.processor.tokenizer.convert_tokens_to_ids("<|latent|>")

        self.base_model.resize_token_embeddings(len(self.processor.tokenizer))
        
        # Initialize the new token embeddings with a known token for stability
        embeddings = self.base_model.get_input_embeddings()
        # Use a similar approach as LLaMA - initialize with "<<" or another token
        try:
            target_id = self.processor.tokenizer.convert_tokens_to_ids("<<")
            if target_id == self.processor.tokenizer.unk_token_id:
                # Fallback to another common token
                target_id = self.processor.tokenizer.convert_tokens_to_ids("think")
                if target_id == self.processor.tokenizer.unk_token_id:
                    target_id = self.processor.tokenizer.convert_tokens_to_ids(".")
        except:
            target_id = self.processor.tokenizer.convert_tokens_to_ids(".")
            
        with torch.no_grad():
            embeddings.weight.data[self.latent_id] = embeddings.weight.data[target_id]

        # Wrap with Coconut
        self.model = QwenVLMCoconut(
            self.base_model,
            self.processor,
            self.latent_id,
            self.processor.tokenizer.eos_token_id,
        )
        self.model.eval()

        # Generation config
        if 'Instruct' in model_path or 'cot' in model_path or 'CoT' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=1)
        kwargs.update(kwargs_default)
        self.kwargs = kwargs

        self.c_thought = c_thought
        
        # Generation kwargs
        kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None)
        kwargs.update(kwargs_default)
        print(f'\nCoconut Qwen Vision - Following kwargs received: {kwargs}, will use as generation config.\n')
        self.kwargs = kwargs

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA', 'MMVet', 'OCRBench'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        if self.c_thought == 0:
            latent_tokens = ''
        else:
            print('<<Thoughts Assigned>>')
            latent_tokens = "<|latent|>" * self.c_thought

        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = (
                f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                f'{latent_tokens}\n'
                f'Respond only with the correct option digit.'
            )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options_str = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options_str}\n'
                f'{latent_tokens}\n'
                f'Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i + 1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}\n{latent_tokens}'
        elif listinstr(['ChartQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'You are provided a chart image and will be asked a question. '
                f'Think step by step through your reasoning process using the latent tokens, '
                f'then provide your final answer.\n'
                f'Question: {question}\n'
                f'{latent_tokens}'
            )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question. '
                f'For yes/no questions, just respond Yes or No. '
                f'If numeric, respond with the number only.\n'
                f'{latent_tokens}\n'
                f'Question: {question}'
            )
        elif listinstr(['MMVet'], dataset):
            self.kwargs['max_new_tokens'] = 1024
            prompt = (
                f'Look at the image and answer the question carefully. '
                f'Use step-by-step reasoning.\n'
                f'Question: {question}\n'
                f'{latent_tokens}'
            )
        elif listinstr(['OCRBench'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully.\n'
                f'{latent_tokens}\n'
                f'Question: {question}'
            )
        else:
            # Default case
            prompt = (
                f'Look at the image and answer the question carefully.\n'
                f'Use step-by-step reasoning and output the final answer.\n'
                f'{question}\n{latent_tokens}'
            )

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def _ensure_image_url(self, image: str) -> str:
        """Ensure image path is a valid URL for qwen_vl_utils."""
        import os
        prefixes = ['http://', 'https://', 'file://', 'data:image;']
        if any(image.startswith(prefix) for prefix in prefixes):
            return image
        if os.path.exists(image):
            return 'file://' + image
        raise ValueError(f'Invalid image: {image}')

    def _prepare_content(self, inputs, dataset=None):
        content = []
        last_text_idx = None
        has_latent = any('<|latent|>' in s['value'] for s in inputs if s['type'] == 'text')
        
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': self._ensure_image_url(s['value'])}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
                last_text_idx = len(content) - 1
        
        # Add latent tokens to the LAST text segment only (if not already present)
        if self.c_thought > 0 and not has_latent and last_text_idx is not None:
            latent_tokens = "<|latent|>" * self.c_thought
            content[last_text_idx]['text'] += latent_tokens
        
        return content

    def generate_inner(self, message, dataset=None):
        """Main generation method using Coconut reasoning."""
        # Build content from message (supports interleaved input)
        content = self._prepare_content(message, dataset=dataset)
        
        # Build messages for Qwen format
        messages = [{'role': 'user', 'content': content}]

        # Process inputs using qwen_vl_utils
        images, videos = self.process_vision_info(messages)
        
        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Process with processor
        inputs = self.processor(
            text=[input_text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors='pt'
        ).to(self.device)

        # Set max tokens based on dataset
        if not self.use_custom_prompt(dataset):
            if dataset is not None and (DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N'):
                self.kwargs['max_new_tokens'] = 512
            else:
                self.kwargs['max_new_tokens'] = 1024

        # Prepare generation kwargs
        generate_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': self.kwargs.get('max_new_tokens', 2048),
        }

        # Add vision-related inputs
        if 'pixel_values' in inputs:
            generate_kwargs['pixel_values'] = inputs['pixel_values']
        if 'image_grid_thw' in inputs:
            generate_kwargs['image_grid_thw'] = inputs['image_grid_thw']



        with torch.no_grad():
            if self.c_thought > 0:
                outputs = self.model.generate(**generate_kwargs)
                generated_text = self.processor.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                return generated_text
            else:
                outputs = self.base_model.generate(**inputs, **self.kwargs)
                
                generated_text = self.processor.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                return generated_text

    def chat_inner(self, message, dataset=None):
        """Chat interface - delegates to generate_inner."""
        return self.generate_inner(message, dataset)

