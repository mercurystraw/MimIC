from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image
from testbed.models.model_base import ModelBase

class LLaMA(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class = AutoTokenizer,
        model_class= AutoModelForCausalLM,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        processor_args = (
            processor_args if processor_args
            else dict(chat_template=self.default_prompt_template)
        )

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )


    @property
    def default_prompt_template(self):
        # fmt: off
        return (
            "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                "{{ messages[0]['role'].capitalize() + ': ' + messages[0]['content'] + '\n'}}"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% set first_role = messages[0]['role'] %}"
            "{% set ns = namespace(generation_role='Assistant') %}"
            "{% for message in messages %}"
                "{% if message['role'] != '' %}"
                    "{{ message['role'].capitalize() }}"
                    "{% if loop.last or loop.nextitem['role'] == first_role %}"
                        "{% set ns.generation_role = message['role'] %}"
                    "{% endif %}"
                    "{% if 'content' in message %}"
                        "{{ ': '}}"
                    "{% else %}"
                        "{{ ':' }}"
                    "{% endif %}"
                "{% endif %}"
                "{% if 'content' in message %}"
                    "{{ ': ' + message['content'] + '\n' }}"
                "{% else %}"
                    "{{ ':' }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ ns.generation_role.capitalize() + ':' }}"
            "{% endif %}"
        )
        # fmt: on

    def process_input(
        self,
        images: Union[List[Image], List[List[Image]]],
        text: Union[
            List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]
        ],
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Processes text and image inputs for the model.

        Args:
            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

            text (Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]):
                A list of texts or a list of lists of texts. For unbatched input, this should be a single-level list
                where each item is either a string or a dictionary. For batched input, this should be a nested list
                (list of lists) where each inner list represents a batch of texts. Dictionaries can follow the
                transformers' conversation format, with keys like "role" and "content".

            prompt_template (str, optional):
                A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        if isinstance(text[0], dict) or (
            isinstance(text[0], list) and isinstance(text[0][0], dict)
        ):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)  # type: ignore[arg-type]

        processor_args = {
            "text": text,
            "padding": kwargs.pop("padding", True),
            "return_tensors":kwargs.pop("return_tensors", "pt"),
            **kwargs,
        }

        if images and any(len(img_batch) > 0 for img_batch in (images if isinstance(images[0], list) else [images])):
            if not hasattr(self.processor, "image_processor"):
                raise ValueError("当前模型不支持图像输入")

            processor_args["images"] = images


        return self.processor(**processor_args)