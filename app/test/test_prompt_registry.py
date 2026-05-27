import unittest

from app.core.load_prompt import load_prompt
from app.evaluation.run_eval import parse_prompt_versions
from app.prompts.prompt_registry import get_prompt_definition, list_prompt_definitions


class PromptRegistryTest(unittest.TestCase):
    def test_get_default_prompt_definition(self):
        definition = get_prompt_definition("answer_out")
        self.assertEqual(definition.prompt_name, "answer_out")
        self.assertEqual(definition.version, "v1")
        self.assertTrue(definition.template_path.endswith("answer_out.prompt"))

    def test_render_prompt_keeps_backward_compatibility(self):
        prompt = load_prompt(
            "answer_out",
            context="ctx",
            history="history",
            item_names="item",
            question="question",
        )
        self.assertIn("ctx", prompt)
        self.assertIn("question", prompt)

    def test_parse_prompt_versions(self):
        result = parse_prompt_versions("answer_out=v1, eval_faithfulness_judge=v1")
        self.assertEqual(
            result,
            {
                "answer_out": "v1",
                "eval_faithfulness_judge": "v1",
            },
        )

    def test_list_prompt_definitions(self):
        definitions = list_prompt_definitions("answer_out")
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0].version, "v1")


if __name__ == "__main__":
    unittest.main()
