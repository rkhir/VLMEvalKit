import torch
from PIL import Image
import os.path as osp
import sys
from vlmeval.vlm.base import BaseModel
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE
import re
import numpy as np
import torch.nn.functional as F
import uuid
import copy
import json
from transformers import StoppingCriteria, StoppingCriteriaList


from vlmeval.vlm.coconut_vlm.coconut import Coconut


class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True
        return False


class StopOnPeriod(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if generated_text.endswith('.'):
            return True
        return False


class CoconutVision(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self,
                 model_path='meta-llama/Llama-3.2-11B-Vision-Instruct',
                 c_thought=2,
                 scheduled_stage=2,
                 max_latent_stage=3,
                 **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using coconut_vision.')
            raise e

        # Load base vision model
        self.base_model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        ).eval()
        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Setup tokenizer and special tokens
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Add Coconut special tokens
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        self.processor.tokenizer.add_tokens(special_tokens)

        self.latent_id = self.processor.tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_id = self.processor.tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_id = self.processor.tokenizer.convert_tokens_to_ids("<|end-latent|>")

        self.base_model.resize_token_embeddings(len(self.processor.tokenizer))
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        embeddings = self.base_model.get_input_embeddings()
        target_id = self.processor.tokenizer.convert_tokens_to_ids("<<")
        for token_id in [self.latent_id, self.start_id, self.end_id]:
            with torch.no_grad():
                embeddings.weight.data[token_id] = embeddings.weight.data[target_id]

        # Wrap with Coconut
        self.model = Coconut(
           self.base_model,
           self.processor,
           self.latent_id,
           self.start_id,
           self.end_id,
           self.processor.tokenizer.eos_token_id
        )
        self.model.eval()
        if 'Instruct' in model_path or 'cot' in model_path or 'CoT' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=1)
        kwargs.update(kwargs_default)
        self.kwargs = kwargs

        self.c_thought = c_thought
        self.scheduled_stage = scheduled_stage
        self.max_latent_stage = max_latent_stage

        # Generation kwargs
        kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None)
        kwargs.update(kwargs_default)
        print(f'Coconut Vision - Following kwargs received: {kwargs}, will use as generation config.')
        self.kwargs = kwargs


    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA', 'MMVet'], dataset):
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

        k = min(self.max_latent_stage, self.scheduled_stage) * self.c_thought
        latent_tokens = f"<|start-latent|>" + "<|latent|>" * k + "<|end-latent|>"

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
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options}\n'
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
        else:
            # Default case - add latent tokens to any question
            prompt = f'Look at the image and answer the question carefully.\n Use step-by-step reasoning and output the final answer. \n{question}\n{latent_tokens}'

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        """Main generation method using Coconut reasoning"""
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        k = min(self.max_latent_stage, self.scheduled_stage) * self.c_thought
        latent_tokens = f"<|start-latent|>" + "<|latent|>" * k + "<|end-latent|>"
        prompt = prompt + latent_tokens

        image = Image.open(image_path)
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]
        # Process inputs
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        # Set max tokens based on dataset
        if not self.use_custom_prompt(dataset):
            if dataset is not None and (DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N'):
                self.kwargs['max_new_tokens'] = 512
            else:
                self.kwargs['max_new_tokens'] = 1024

        seq_len = inputs['input_ids'].shape[1]
        inputs['position_ids'] = torch.arange(seq_len, device=self.device).unsqueeze(0)

        generate_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
        }

        if 'aspect_ratio_ids' in inputs and inputs['aspect_ratio_ids'] is not None:
            generate_kwargs['aspect_ratio_ids'] = inputs['aspect_ratio_ids']
            if 'aspect_ratio_mask' in inputs and inputs['aspect_ratio_mask'] is not None:
                generate_kwargs['aspect_ratio_mask'] = inputs['aspect_ratio_mask']
        else:
            # Generate aspect_ratio_ids and aspect_ratio_mask if not provided by processor
            # For mllama models, we need to provide both when pixel_values are present
            batch_size, max_num_images, max_num_tiles = inputs['pixel_values'].shape[:3]

            # Default to aspect ratio id 1 (1:1 ratio, offset by 1 as per mllama docs)
            generate_kwargs['aspect_ratio_ids'] = torch.ones((batch_size, max_num_images), dtype=torch.long, device=self.device)

            # Create aspect_ratio_mask - for single tile images, mask the first tile only
            generate_kwargs['aspect_ratio_mask'] = torch.zeros((batch_size, max_num_images, max_num_tiles), dtype=torch.long, device=self.device)
            generate_kwargs['aspect_ratio_mask'][:, :, 0] = 1  # Enable first tile for all images

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.kwargs)
            self.kwargs['max_new_tokens']=300

        generated_text = self.processor.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return generated_text

    def chat_inner(self, message, dataset=None):
        """Chat interface - delegates to generate_inner"""
        return self.generate_inner(message, dataset)
