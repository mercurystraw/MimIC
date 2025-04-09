from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from testbed.models.model_base import ModelBase

class Mistral(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoTokenizer,
        model_class=AutoModelForCausalLM,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        processor_args = (
            processor_args
            if processor_args
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
        # fmt: off   代码格式化控制注释：希望保留特定代码段的手动排版格式时;防止代码格式化工具（如 Black）自动修改代码布局
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
