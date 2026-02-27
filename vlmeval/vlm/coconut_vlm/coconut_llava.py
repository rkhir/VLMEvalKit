"""
Coconut Vision wrapper for LLaVA / LLaVA-Next models.

Implements Meta's Coconut continuous thought approach for LLaVA models,
enabling latent reasoning in continuous embedding space before generating
the final text response.

Supports:
- LlavaNextForConditionalGeneration (LLaVA v1.6, default)
- LlavaForConditionalGeneration (LLaVA v1.5)
- LlavaOnevisionForConditionalGeneration (LLaVA-OneVision)
"""

import torch
import logging
import string

from vlmeval.vlm.base import BaseModel
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE

from vlmeval.vlm.coconut_vlm.llava_vlmcoconut import LLaVAVLMCoconut


class CoconutLLaVA(BaseModel):
    """
    Coconut-enhanced LLaVA model for continuous latent reasoning.

    Wraps a HuggingFace LLaVA model with the Coconut mechanism:
    <|latent|> tokens are inserted into the prompt, and the model performs
    multiple forward passes that feed the last hidden state back as the
    latent token embedding — allowing "thinking" in continuous space
    before generating the answer.
    """

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path='llava-hf/llava-v1.6-vicuna-7b-hf',
        c_thought=2,
        **kwargs,
    ):
        try:
            from transformers import AutoProcessor
        except Exception as e:
            logging.critical(
                'Please install transformers>=4.45.0 before using CoconutLLaVA.'
            )
            raise e

        self.model_path = model_path

        # ── load base LLaVA model ────────────────────────────────────────
        flash_attn_flag = False
        try:
            import flash_attn  # noqa: F401
            flash_attn_flag = True
        except ImportError:
            pass

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='auto',
        )
        if flash_attn_flag:
            load_kwargs['attn_implementation'] = 'flash_attention_2'

        self.base_model = self._load_model(model_path, load_kwargs)
        self.device = 'cuda'

        # ── load processor ───────────────────────────────────────────────
        self.processor = AutoProcessor.from_pretrained(model_path)

        # ── tokenizer setup ──────────────────────────────────────────────
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add Coconut special token
        special_tokens = ["<|latent|>"]
        tokenizer.add_tokens(special_tokens)
        self.latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

        self.base_model.resize_token_embeddings(len(tokenizer))

        # Initialize latent token embedding with a known token for stability
        embeddings = self.base_model.get_input_embeddings()
        target_id = self._find_init_token_id(tokenizer)
        with torch.no_grad():
            embeddings.weight.data[self.latent_id] = (
                embeddings.weight.data[target_id]
            )

        # ── wrap with Coconut ────────────────────────────────────────────
        self.model = LLaVAVLMCoconut(
            self.base_model,
            self.processor,
            self.latent_id,
            tokenizer.eos_token_id,
        )
        self.model.eval()

        self.c_thought = c_thought

        # Generation kwargs
        kwargs_default = dict(
            do_sample=False,
            max_new_tokens=2048,
            temperature=0.0,
            top_p=None,
        )
        kwargs.update(kwargs_default)
        print(
            f'\nCoconut LLaVA - Following kwargs received: {kwargs}, '
            f'will use as generation config.\n'
        )
        self.kwargs = kwargs

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(model_path, load_kwargs):
        """Load the appropriate HuggingFace LLaVA model class."""
        path_lower = model_path.lower()

        if 'onevision' in path_lower:
            from transformers import LlavaOnevisionForConditionalGeneration
            return LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            ).eval()

        # Try LlavaNext first (covers v1.6 and newer)
        try:
            from transformers import LlavaNextForConditionalGeneration
            return LlavaNextForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            ).eval()
        except Exception:
            pass

        # Fallback to basic LlavaForConditionalGeneration (v1.5)
        from transformers import LlavaForConditionalGeneration
        return LlavaForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        ).eval()

    @staticmethod
    def _find_init_token_id(tokenizer):
        """Find a reasonable token to initialize the <|latent|> embedding."""
        for candidate in ["<<", "think", "."]:
            try:
                tid = tokenizer.convert_tokens_to_ids(candidate)
                if tid != tokenizer.unk_token_id:
                    return tid
            except Exception:
                continue
        return tokenizer.convert_tokens_to_ids(".")

    # ── prompt building ──────────────────────────────────────────────────

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(
            ['AI2D', 'MMMU', 'MathVista', 'ChartQA',
             'DocVQA', 'MMVet', 'OCRBench'],
            dataset,
        ):
            return True
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

        latent_tokens = (
            '' if self.c_thought == 0
            else '<|latent|>' * self.c_thought
        )

        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = (
                f'Look at the scientific diagram carefully and answer the '
                f'following question: {question}\n'
                f'{latent_tokens}\n'
                f'Respond only with the correct option digit.'
            )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options_str = '\n'.join(
                [f'{key}. {item}' for key, item in options.items()]
            )
            prompt = (
                f'Look at the image carefully and solve the following '
                f'question step-by-step. '
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
                f'You are provided a chart image and will be asked a '
                f'question. Think step by step through your reasoning '
                f'process using the latent tokens, then provide your '
                f'final answer.\n'
                f'Question: {question}\n'
                f'{latent_tokens}'
            )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the '
                f'question. For yes/no questions, just respond Yes or No. '
                f'If numeric, respond with the number only.\n'
                f'Question: {question}'
                f'{latent_tokens}\n'
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
                f'Question: {question}'
                f'{latent_tokens}\n'
            )
        else:
            prompt = (
                f'Look at the image and answer the question carefully.\n'
                f'Use step-by-step reasoning and output the final answer.\n'
                f'{question}\n{latent_tokens}'
            )

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    # ── output cleanup ───────────────────────────────────────────────────

    @staticmethod
    def _output_process(answer):
        """Strip common prompt / role artefacts from generated text."""
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        elif '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()
        elif '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()

        answer = answer.replace('<unk>', '')
        return answer

    # ── generation ───────────────────────────────────────────────────────

    def generate_inner(self, message, dataset=None):
        """Main generation method using Coconut reasoning."""

        # Build conversation content for HF chat template
        content, images = [], []
        for msg in message:
            if msg['type'] == 'text':
                content.append({'type': 'text', 'text': msg['value']})
            elif msg['type'] == 'image':
                content.append({'type': 'image'})
                images.append(Image.open(msg['value']).convert('RGB'))

        conversation = [{'role': 'user', 'content': content}]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Process inputs (tokenise + image preprocessing)
        inputs = self.processor(
            prompt,
            images if images else None,
            return_tensors='pt',
        ).to(self.device)

        # Adjust max_new_tokens for dataset type
        if not self.use_custom_prompt(dataset):
            if dataset is not None and DATASET_TYPE(dataset) in ('MCQ', 'Y/N'):
                self.kwargs['max_new_tokens'] = 512
            else:
                self.kwargs['max_new_tokens'] = 1024

        # Collect kwargs for the Coconut wrapper
        generate_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': self.kwargs.get('max_new_tokens', 2048),
        }
        if 'pixel_values' in inputs:
            generate_kwargs['pixel_values'] = inputs['pixel_values']
        if 'image_sizes' in inputs:
            generate_kwargs['image_sizes'] = inputs['image_sizes']

        with torch.no_grad():
            if self.c_thought > 0:
                outputs = self.model.generate(**generate_kwargs)
                generated_text = self.processor.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                return generated_text

            # No latent tokens → use base model directly
            outputs = self.base_model.generate(**inputs, **self.kwargs)
            generated_text = self.processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            return generated_text

    def chat_inner(self, message, dataset=None):
        """Chat interface — delegates to generate_inner."""
        return self.generate_inner(message, dataset)
