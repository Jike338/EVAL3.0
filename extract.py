import re
from models.gpt import GPT
from prompts import demo_prompt_mathvista, demo_prompt_superclevr_counting
from tqdm import tqdm

class Extractor:
    def __init__(self, 
                 items_with_raw_responses, 
                 args=None, 
                 use_quick_extract_w_gpt=True, 
                 use_gpt_extract=False, 
                 use_answer_tag_extract=False):
        self.items_with_raw_responses = items_with_raw_responses
        self.args = args
        self.use_quick_extract_w_gpt = use_quick_extract_w_gpt
        self.use_gpt_extract = use_gpt_extract
        self.use_answer_tag_extract = use_answer_tag_extract
        self.gpt_extractor = GPT()
        
        if self.args.gen_prompt_suffix_type == "cot_tag":
            self.use_quick_extract_w_gpt = False
            self.use_answer_tag_extract = True
        
    def _save(self):
        from utils import save_response_to_json
        save_response_to_json(self.args, self.items_with_raw_responses)

    def _build_gpt_extraction_prompt(self, question_prompt, resp):
        demo_prompt = self.demo_prompt.strip()
        test_prompt = f"{question_prompt}\n\n{resp}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt

    def _extract_with_gpt(self, question_prompt, resp):
        print("using GPT extract")
        full_prompt = self._build_gpt_extraction_prompt(question_prompt, resp)
        return self.gpt_extractor.extract_answer_from_raw_response(full_prompt)

    def _is_simple_response(self, resp):
        return (
            re.fullmatch(r"[A-Z]", resp) or              
            re.fullmatch(r"\d+(?:\.\d+)?", resp)          
        )

    def _extract_answer_content(self, resp):
        # Extract anything within <answer>...</answer> tags
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(answer_pattern, resp, re.DOTALL)
        
        if match:
            return match.group(1)
        return None

    def _extract(self, question_prompt, raw_responses):
        extracted_responses = []
        for resp in raw_responses:
            if self.use_quick_extract_w_gpt:
                resp = resp.strip()
                if not self._is_simple_response(resp):
                    resp = self._extract_with_gpt(question_prompt, resp)
            elif self.use_answer_tag_extract:
                resp = self._extract_answer_content(resp)
            elif self.use_gpt_extract:
                resp = self._extract_with_gpt(question_prompt, resp)
            else:
                raise NotImplementedError(f"No extraction method selected.")
            extracted_responses.append(resp)
        return extracted_responses
    
    def _extract_raw_response(self):
        for item in tqdm(self.items_with_raw_responses, desc="Extracting answers"):
            if 'extracted_response' in item:
                print("already exists")
                continue
            raw_responses = item['raw_response']
            question_prompt = item['question_prompt']
            extracted_responses = self._extract(question_prompt, raw_responses)
            item['extracted_response'] = extracted_responses
    
    def extract_ans_and_save(self):
        if "mathvista" in self.args.task_name.lower():
            self.demo_prompt = demo_prompt_mathvista
        if "superclevr_counting" in self.args.task_name.lower():
            self.demo_prompt = demo_prompt_superclevr_counting
        else:
            raise NotImplementedError(f"No extractor defined for task: {self.args.task_name}")
        
        self._extract_raw_response()
        self.args.duty_type = "extract"
        self._save()
